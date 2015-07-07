# (C) British Crown Copyright 2013 - 2015, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Statistical operations between cubes.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import numpy as np
import numpy.ma as ma
import iris
from iris.util import broadcast_to_shape


def _ones_like(cube):
    """
    Return a copy of cube with the same mask, but all data values set to 1.
    """
    ones_cube = cube.copy()
    ones_cube.data = np.ones_like(cube.data)
    ones_cube.rename('unknown')
    ones_cube.units = 1
    return ones_cube


def pearsonr(cube_a, cube_b, corr_coords=None, weights=None, mdtol=1.,
             common_mask=False):
    """
    Calculate the Pearson's r correlation coefficient over specified
    dimensions.

    Args:

    * cube_a, cube_b (cubes):
        Cubes between which the correlation will be calculated.  The cubes
        should either be the same shape and have the same dimension coordinates
        or one cube should be broadcastable to the other.
    * corr_coords (str or list of str):
        The cube coordinate name(s) over which to calculate correlations. If no
        names are provided then correlation will be calculated over all common
        cube dimensions.
    * weights (numpy.ndarray, optional):
        Weights array of same shape as (the smaller of) cube_a and cube_b. Note
        that latitude/longitude area weights can be calculated using
        :func:`iris.analysis.cartography.area_weights`.
    * mdtol (float, optional):
        Tolerance of missing data. The missing data fraction is calculated
        based on the number of grid cells masked in both cube_a and cube_b. If
        this fraction exceed mdtol, the returned value in the corresponding
        cell is masked. mdtol=0 means no missing data is tolerated while
        mdtol=1 means the resulting element will be masked if and only if all
        contributing elements are masked in cube_a or cube_b. Defaults to 1.
    * common_mask (bool):
        If True, applies a common mask to cube_a and cube_b so only cells which
        are unmasked in both cubes contribute to the calculation. If False, the
        variance for each cube is calculated from all available cells. Defaults
        to False.

    Returns:
        A cube of the correlation between the two input cubes along the
        specified dimensions, at each point in the remaining dimensions of the
        cubes.

        For example providing two time/altitude/latitude/longitude cubes and
        corr_coords of 'latitude' and 'longitude' will result in a
        time/altitude cube describing the latitude/longitude (i.e. pattern)
        correlation at each time/altitude point.

    Reference:
        http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    """

    # Assign larger cube to cube_1
    if cube_b.ndim > cube_a.ndim:
        cube_1 = cube_b
        cube_2 = cube_a
    else:
        cube_1 = cube_a
        cube_2 = cube_b

    dim_coords_1 = [coord.name() for coord in cube_1.dim_coords]
    dim_coords_2 = [coord.name() for coord in cube_2.dim_coords]
    common_dim_coords = list(set(dim_coords_1) & set(dim_coords_2))
    # If no coords passed then set to all common dimcoords of cubes.
    if corr_coords is None:
        corr_coords = common_dim_coords

    smaller_shape = cube_2.shape

    # Match up data masks if required.
    if common_mask:
        # Create a cube of 1's with a common mask.
        if ma.is_masked(cube_2.data):
            mask_cube = _ones_like(cube_2)
        else:
            mask_cube = 1.
        if ma.is_masked(cube_1.data):
            # Take a slice to avoid unnecessary broadcasting of cube_2.
            slice_coords = [dim_coords_1[i] for i in range(cube_1.ndim) if
                            dim_coords_1[i] not in common_dim_coords and
                            np.array_equal(cube_1.data.mask.any(axis=i),
                                           cube_1.data.mask.all(axis=i))]
            cube_1_slice = cube_1.slices_over(slice_coords).next()
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
            raise ValueError("weights array should have dimensions {}".
                             format(smaller_shape))

        dims_1_common = [i for i in range(cube_1.ndim) if
                         dim_coords_1[i] in common_dim_coords]
        weights_1 = broadcast_to_shape(weights, cube_1.shape, dims_1_common)
        if cube_2.shape != smaller_shape:
            dims_2_common = [i for i in range(cube_2.ndim) if
                             dim_coords_2[i] in common_dim_coords]
            weights_2 = broadcast_to_shape(weights, cube_2.shape,
                                           dims_2_common)
        else:
            weights_2 = weights

    # Calculate correlations.
    s1 = cube_1 - cube_1.collapsed(corr_coords, iris.analysis.MEAN,
                                   weights=weights_1)
    s2 = cube_2 - cube_2.collapsed(corr_coords, iris.analysis.MEAN,
                                   weights=weights_2)

    covar = (s1*s2).collapsed(corr_coords, iris.analysis.SUM,
                              weights=weights_1, mdtol=mdtol)
    var_1 = (s1**2).collapsed(corr_coords, iris.analysis.SUM,
                              weights=weights_1)
    var_2 = (s2**2).collapsed(corr_coords, iris.analysis.SUM,
                              weights=weights_2)

    denom = iris.analysis.maths.apply_ufunc(np.sqrt, var_1*var_2,
                                            new_unit=covar.units)
    corr_cube = covar / denom
    corr_cube.rename("Pearson's r")

    return corr_cube
