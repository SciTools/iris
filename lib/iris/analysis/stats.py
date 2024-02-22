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
    """Calculate the Pearson's r correlation coefficient over specified dimensions.

    Parameters
    ----------
    cube_a, cube_b : :class:`iris.cube.Cube`
        Cubes between which the correlation will be calculated.  The cubes
        should either be the same shape and have the same dimension coordinates
        or one cube should be broadcastable to the other.  Broadcasting rules
        are the same as those for cube arithmetic (see :ref:`cube maths`).
    corr_coords : str or list of str, optional
        The cube coordinate name(s) over which to calculate correlations. If no
        names are provided then correlation will be calculated over all common
        cube dimensions.
    weights : :class:`numpy.ndarray`, optional
        Weights array of same shape as (the smaller of) `cube_a` and `cube_b`.
        Note that latitude/longitude area weights can be calculated using
        :func:`iris.analysis.cartography.area_weights`.
    mdtol : float, default=1.0
        Tolerance of missing data. The missing data fraction is calculated
        based on the number of grid cells masked in both `cube_a` and `cube_b`.
        If this fraction exceed `mdtol`, the returned value in the
        corresponding cell is masked. `mdtol` =0 means no missing data is
        tolerated while `mdtol` =1 means the resulting element will be masked
        if and only if all contributing elements are masked in `cube_a` or
        `cube_b`.
    common_mask : bool, default=False
        If ``True``, applies a common mask to cube_a and cube_b so only cells
        which are unmasked in both cubes contribute to the calculation. If
        ``False``, the variance for each cube is calculated from all available
        cells.

    Returns
    -------
    :class:`~iris.cube.Cube`
        A cube of the correlation between the two input cubes along the
        specified dimensions, at each point in the remaining dimensions of the
        cubes.

        For example providing two time/altitude/latitude/longitude cubes and
        `corr_coords` of 'latitude' and 'longitude' will result in a
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
    lhs_cube_resolved = resolver.lhs_cube_resolved
    rhs_cube_resolved = resolver.rhs_cube_resolved

    if lhs_cube_resolved.has_lazy_data() or rhs_cube_resolved.has_lazy_data():
        al = da
        array_lhs = lhs_cube_resolved.lazy_data()
        array_rhs = rhs_cube_resolved.lazy_data()
    else:
        al = np
        array_lhs = lhs_cube_resolved.data
        array_rhs = rhs_cube_resolved.data

    # If no coords passed then set to all common dimcoords of cubes.
    if corr_coords is None:
        dim_coords_1 = {coord.name() for coord in lhs_cube_resolved.dim_coords}
        dim_coords_2 = {coord.name() for coord in rhs_cube_resolved.dim_coords}
        corr_coords = list(dim_coords_1.intersection(dim_coords_2))

    # Interpret coords as array dimensions.
    corr_dims = set()
    if isinstance(corr_coords, str):
        corr_coords = [corr_coords]
    for coord in corr_coords:
        corr_dims.update(lhs_cube_resolved.coord_dims(coord))

    corr_dims = tuple(corr_dims)

    # Match up data masks if required.
    if common_mask:
        mask_lhs = al.ma.getmaskarray(array_lhs)
        if al is np:
            # Reduce all invariant dimensions of mask_lhs to length 1.  This avoids
            # unnecessary broadcasting of array_rhs.
            index = tuple(
                slice(0, 1)
                if np.array_equal(mask_lhs.any(axis=dim), mask_lhs.all(axis=dim))
                else slice(None)
                for dim in range(mask_lhs.ndim)
            )
            mask_lhs = mask_lhs[index]

        array_rhs = _mask_array(array_rhs, mask_lhs)
        array_lhs = _mask_array(array_lhs, al.ma.getmaskarray(array_rhs))

    # Broadcast weights to shape of arrays if necessary.
    if weights is None:
        weights_lhs = weights_rhs = None
    else:
        if weights.shape != smaller_shape:
            msg = f"weights array should have dimensions {smaller_shape}"
            raise ValueError(msg)

        wt_resolver = Resolve(cube_1, cube_2.copy(weights))
        weights = wt_resolver.rhs_cube_resolved.data
        weights_rhs = np.broadcast_to(weights, array_rhs.shape)
        weights_lhs = np.broadcast_to(weights, array_lhs.shape)

    # Calculate correlations.
    s_lhs = array_lhs - al.ma.average(
        array_lhs, axis=corr_dims, weights=weights_lhs, keepdims=True
    )
    s_rhs = array_rhs - al.ma.average(
        array_rhs, axis=corr_dims, weights=weights_rhs, keepdims=True
    )

    s_prod = resolver.cube(s_lhs * s_rhs)

    # Use cube collapsed method as it takes care of coordinate collapsing and missing
    # data tolerance.
    covar = s_prod.collapsed(
        corr_coords, iris.analysis.SUM, weights=weights_lhs, mdtol=mdtol
    )

    var_lhs = iris.analysis._sum(s_lhs**2, axis=corr_dims, weights=weights_lhs)
    var_rhs = iris.analysis._sum(s_rhs**2, axis=corr_dims, weights=weights_rhs)

    denom = np.sqrt(var_lhs * var_rhs)

    corr_cube = covar / denom
    corr_cube.rename("Pearson's r")
    corr_cube.units = 1

    return corr_cube
