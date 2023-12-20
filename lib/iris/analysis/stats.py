# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Statistical operations between cubes."""

import dask.array as da
import numpy as np

import iris
from iris.common import SERVICES, Resolve
from iris.common.lenient import _lenient_client
from iris.util import _mask_array


@_lenient_client(services=SERVICES)
def pearsonr(
    cube_a,
    cube_b,
    corr_coords=None,
    weights=None,
    mdtol=1.0,
    common_mask=False,
):
    """Calculate the Pearson's r correlation coefficient over specified
    dimensions.

    Parameters
    ----------
    cube_a, cube_b : cubes
        Cubes between which the correlation will be calculated.  The cubes
        should either be the same shape and have the same dimension coordinates
        or one cube should be broadcastable to the other.
    corr_coords : str or list of str
        The cube coordinate name(s) over which to calculate correlations. If no
        names are provided then correlation will be calculated over all common
        cube dimensions.
    weights : :class:`numpy.ndarray`, optional
        Weights array of same shape as (the smaller of) cube_a and cube_b. Note
        that latitude/longitude area weights can be calculated using
        :func:`iris.analysis.cartography.area_weights`.
    mdtol : float, default=1.0
        Tolerance of missing data. The missing data fraction is calculated
        based on the number of grid cells masked in both cube_a and cube_b. If
        this fraction exceed mdtol, the returned value in the corresponding
        cell is masked. mdtol=0 means no missing data is tolerated while
        mdtol=1 means the resulting element will be masked if and only if all
        contributing elements are masked in cube_a or cube_b. Defaults to 1.
    common_mask : bool, default=False
        If True, applies a common mask to cube_a and cube_b so only cells which
        are unmasked in both cubes contribute to the calculation. If False, the
        variance for each cube is calculated from all available cells. Defaults
        to False.

    Returns
    -------
    :class:`~iris.cube.Cube`
        A cube of the correlation between the two input cubes along the
        specified dimensions, at each point in the remaining dimensions of the
        cubes.

        For example providing two time/altitude/latitude/longitude cubes and
        corr_coords of 'latitude' and 'longitude' will result in a
        time/altitude cube describing the latitude/longitude (i.e. pattern)
        correlation at each time/altitude point.

    Notes
    -----
    If either of the input cubes has lazy data, the result will have lazy data.

    Reference:
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    """
    # Assign larger cube to cube_1 for simplicity.
    if cube_b.ndim > cube_a.ndim:
        cube_1 = cube_b
        cube_2 = cube_a
    else:
        cube_1 = cube_a
        cube_2 = cube_b

    smaller_shape = cube_2.shape

    # Get the broadcast, auto-transposed safe versions of the cube operands.
    resolver = Resolve(cube_1, cube_2)
    cube_1 = resolver.lhs_cube_resolved
    cube_2 = resolver.rhs_cube_resolved

    if cube_1.has_lazy_data() or cube_2.has_lazy_data():
        al = da
        array_1 = cube_1.lazy_data()
        array_2 = cube_2.lazy_data()
    else:
        al = np
        array_1 = cube_1.data
        array_2 = cube_2.data

    # If no coords passed then set to all common dimcoords of cubes.
    if corr_coords is None:
        dim_coords_1 = {coord.name() for coord in cube_1.dim_coords}
        dim_coords_2 = {coord.name() for coord in cube_2.dim_coords}
        corr_coords = list(dim_coords_1.intersection(dim_coords_2))

    # Interpret coords as array dimensions.
    corr_dims = set()
    if isinstance(corr_coords, str):
        corr_coords = [corr_coords]
    for coord in corr_coords:
        corr_dims.update(cube_1.coord_dims(coord))

    corr_dims = tuple(corr_dims)

    # Match up data masks if required.
    if common_mask:
        mask_1 = al.ma.getmaskarray(array_1)
        if al is np:
            # Reduce all invariant dimensions of mask_1 to length 1.  This avoids
            # unnecessary broadcasting of array_2.
            index = tuple(
                slice(0, 1)
                if np.array_equal(mask_1.any(axis=dim), mask_1.all(axis=dim))
                else slice(None)
                for dim in range(mask_1.ndim)
            )
            mask_1 = mask_1[index]

        array_2 = _mask_array(array_2, mask_1)
        array_1 = _mask_array(array_1, al.ma.getmaskarray(array_2))

    # Broadcast weights to shape of arrays if necessary.
    if weights is None:
        weights_1 = weights_2 = None
    else:
        if weights.shape != smaller_shape:
            msg = f"weights array should have dimensions {smaller_shape}"
            raise ValueError(msg)

        if resolver.reorder_src_dims is not None:
            # Apply same transposition as was done to cube_2 within Resolve.
            weights = weights.transpose(resolver.reorder_src_dims)

        # Reshape to add in any length-1 dimensions needed for broadcasting.
        weights = weights.reshape(cube_2.shape)

        weights_2 = np.broadcast_to(weights, array_2.shape)
        weights_1 = np.broadcast_to(weights, array_1.shape)

    # Calculate correlations.
    s1 = array_1 - al.ma.average(
        array_1, axis=corr_dims, weights=weights_1, keepdims=True
    )
    s2 = array_2 - al.ma.average(
        array_2, axis=corr_dims, weights=weights_2, keepdims=True
    )

    s_prod = resolver.cube(s1 * s2)

    # Use cube collapsed method as it takes care of coordinate collapsing and missing
    # data tolerance.
    covar = s_prod.collapsed(
        corr_coords, iris.analysis.SUM, weights=weights_1, mdtol=mdtol
    )

    var_1 = iris.analysis._sum(s1**2, axis=corr_dims, weights=weights_1)
    var_2 = iris.analysis._sum(s2**2, axis=corr_dims, weights=weights_2)

    denom = np.sqrt(var_1 * var_2)

    corr_cube = covar / denom
    corr_cube.rename("Pearson's r")
    corr_cube.units = 1

    return corr_cube
