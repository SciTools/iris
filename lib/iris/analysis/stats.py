# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Statistical operations between cubes."""

import numpy as np
import numpy.ma as ma

import iris
from iris.util import broadcast_to_shape


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
    Reference:
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    This operation is non-lazy.

    """
    # Assign larger cube to cube_1
    if cube_b.ndim > cube_a.ndim:
        cube_1 = cube_b
        cube_2 = cube_a
    else:
        cube_1 = cube_a
        cube_2 = cube_b

    smaller_shape = cube_2.shape

    dim_coords_1 = [coord.name() for coord in cube_1.dim_coords]
    dim_coords_2 = [coord.name() for coord in cube_2.dim_coords]
    common_dim_coords = list(set(dim_coords_1) & set(dim_coords_2))
    # If no coords passed then set to all common dimcoords of cubes.
    if corr_coords is None:
        corr_coords = common_dim_coords

    def _ones_like(cube):
        # Return a copy of cube with the same mask, but all data values set to 1.
        # The operation is non-lazy.
        # For safety we also discard any cell-measures and ancillary-variables, to
        # avoid cube arithmetic possibly objecting to them, or inadvertently retaining
        # them in the result where they might be inappropriate.
        ones_cube = cube.copy()
        ones_cube.data = np.ones_like(cube.data)
        ones_cube.rename("unknown")
        ones_cube.units = 1
        for cm in ones_cube.cell_measures():
            ones_cube.remove_cell_measure(cm)
        for av in ones_cube.ancillary_variables():
            ones_cube.remove_ancillary_variable(av)
        return ones_cube

    # Match up data masks if required.
    if common_mask:
        # Create a cube of 1's with a common mask.
        if ma.is_masked(cube_2.data):
            mask_cube = _ones_like(cube_2)
        else:
            mask_cube = 1.0
        if ma.is_masked(cube_1.data):
            # Take a slice to avoid unnecessary broadcasting of cube_2.
            slice_coords = [
                dim_coords_1[i]
                for i in range(cube_1.ndim)
                if dim_coords_1[i] not in common_dim_coords
                and np.array_equal(
                    cube_1.data.mask.any(axis=i), cube_1.data.mask.all(axis=i)
                )
            ]
            cube_1_slice = next(cube_1.slices_over(slice_coords))
            mask_cube = _ones_like(cube_1_slice) * mask_cube
        # Apply common mask to data.
        if isinstance(mask_cube, iris.cube.Cube):
            cube_1 = cube_1 * mask_cube
            cube_2 = mask_cube * cube_2
            dim_coords_2 = [coord.name() for coord in cube_2.dim_coords]

    # Broadcast weights to shape of cubes if necessary.
    if weights is None or cube_1.shape == smaller_shape:
        weights_1 = weights
        weights_2 = weights
    else:
        if weights.shape != smaller_shape:
            raise ValueError(
                "weights array should have dimensions {}".format(smaller_shape)
            )

        dims_1_common = [
            i for i in range(cube_1.ndim) if dim_coords_1[i] in common_dim_coords
        ]
        weights_1 = broadcast_to_shape(weights, cube_1.shape, dims_1_common)
        if cube_2.shape != smaller_shape:
            dims_2_common = [
                i for i in range(cube_2.ndim) if dim_coords_2[i] in common_dim_coords
            ]
            weights_2 = broadcast_to_shape(weights, cube_2.shape, dims_2_common)
        else:
            weights_2 = weights

    # Calculate correlations.
    s1 = cube_1 - cube_1.collapsed(corr_coords, iris.analysis.MEAN, weights=weights_1)
    s2 = cube_2 - cube_2.collapsed(corr_coords, iris.analysis.MEAN, weights=weights_2)

    covar = (s1 * s2).collapsed(
        corr_coords, iris.analysis.SUM, weights=weights_1, mdtol=mdtol
    )
    var_1 = (s1**2).collapsed(corr_coords, iris.analysis.SUM, weights=weights_1)
    var_2 = (s2**2).collapsed(corr_coords, iris.analysis.SUM, weights=weights_2)

    denom = iris.analysis.maths.apply_ufunc(
        np.sqrt, var_1 * var_2, new_unit=covar.units
    )
    corr_cube = covar / denom
    corr_cube.rename("Pearson's r")

    return corr_cube
