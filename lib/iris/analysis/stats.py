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

import numpy as np

import iris


def _get_calc_view(cube_a, cube_b, corr_coords):
    """
    This function takes two cubes and returns cubes which are
    flattened so that efficient comparisons can be performed
    between the two. If the arrays are maksed then only values
    that are unmasked in both arrays are used.

    Args:

    * cube_a:
        First cube of data
    * cube_b:
        Second cube of data, compatible with cube_a
    * corr_coords:
        Names of the dimension coordinates over which
        to calculate correlations.

    Returns:

    * reshaped_a/reshaped_b:
        The data arrays of cube_a/cube_b reshaped
        so that the dimensions to be compared are
        flattened into the 0th dimension of the
        return array and other dimensions are
        preserved.
    * res_ind:
        The indices of the dimensions that we
        are not comparing, in terms of cube_a/cube_b

    """

    # Following lists to be filled with:
    # indices of dimension we are not comparing
    res_ind = []
    # indices of dimensions we are comparing
    slice_ind = []
    for i, c in enumerate(cube_a.dim_coords):
        if not c.name() in corr_coords:
            res_ind.append(i)
        else:
            slice_ind.append(i)

    # sanitise input
    dim_coord_names = [c.name() for c in cube_a.dim_coords]
    if corr_coords is None:
        corr_coords = dim_coord_names

    if ([c.name() for c in cube_a.dim_coords] !=
            [c.name() for c in cube_b.dim_coords]):
        raise ValueError("Cubes are incompatible.")

    for c in corr_coords:
        if c not in dim_coord_names:
            raise ValueError("%s coord "
                             "does not exist in cube." % c)

    # Reshape data to be data to correlate in 0th dim and
    # other grid points in 1st dim.
    # Transpose to group the correlation data dims before the
    # grid point dims.
    data_a = cube_a.data.view()
    data_b = cube_b.data.view()
    dim_i_len = np.prod(np.array(cube_a.shape)[slice_ind])
    dim_j_len = np.prod(np.array(cube_a.shape)[res_ind])
    reshaped_a = data_a.transpose(slice_ind+res_ind)\
                       .reshape(dim_i_len, dim_j_len)
    reshaped_b = data_b.transpose(slice_ind+res_ind)\
                       .reshape(dim_i_len, dim_j_len)

    # Remove data where one or both cubes are masked
    # First deal with the case that either cube is unmasked
    # Collapse masks to the dimension we are correlating over (0th)
    if np.ma.is_masked(reshaped_a):
        a_not_masked = np.logical_not(reshaped_a.mask).any(axis=1)
    else:
        a_not_masked = True
    if np.ma.is_masked(reshaped_b):
        b_not_masked = np.logical_not(reshaped_b.mask.any(axis=1))
    else:
        b_not_masked = True

    both_not_masked = a_not_masked & b_not_masked
    try:
        # compress to good values using mask array
        return_a = reshaped_a.compress(both_not_masked)
        return_b = reshaped_b.compress(both_not_masked)
    except ValueError:
        # expect when masks are just non-array True/False
        return_a = reshaped_a
        return_b = reshaped_b

    return return_a, return_b, res_ind


def pearsonr(cube_a, cube_b, corr_coords=None):
    """
    Calculates the n-D Pearson's r correlation
    cube over the dimensions associated with the
    given coordinates.

    Returns a cube of the correlation between the two
    cubes along the dimensions of the given
    coordinates, at each point in the remaining
    dimensions of the cubes.

    For example providing two time/altitude/latitude/longitude
    cubes and corr_coords of 'latitude' and 'longitude' will result
    in a time/altitude cube describing the latitude/longitude
    (i.e. pattern) correlation at each time/altitude point.

    Args:

    * cube_a, cube_b (cubes):
        Between which the correlation field will be calculated.
        Cubes should be the same shape and have the
        same dimension coordinates.
    * corr_coords (list of str):
        The cube coordinate names over which to calculate
        correlations. If no names are provided then
        correlation will be calculated over all cube
        dimensions.

    Returns:
        Cube of correlations.

    Reference:
        http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    """

    # If no coords passed then set to all coords of cube.
    if corr_coords is None:
        corr_coords = [c.name() for c in cube_a.dim_coords]

    vec_a, vec_b, res_ind = _get_calc_view(cube_a,
                                           cube_b,
                                           corr_coords)

    sa = vec_a - np.mean(vec_a, 0)
    sb = vec_b - np.mean(vec_b, 0)
    flat_corrs = np.sum((sa*sb), 0)/np.sqrt(np.sum(sa**2, 0)*np.sum(sb**2, 0))

    corrs = flat_corrs.reshape([cube_a.shape[i] for i in res_ind])

    # Construct cube to hold correlation results.
    corrs_cube = iris.cube.Cube(corrs)
    corrs_cube.long_name = "Pearson's r"
    corrs_cube.units = "1"
    for i, dim in enumerate(res_ind):
        c = cube_a.dim_coords[dim]
        corrs_cube.add_dim_coord(c, i)

    return corrs_cube
