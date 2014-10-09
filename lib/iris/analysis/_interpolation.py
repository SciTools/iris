# (C) British Crown Copyright 2014, Met Office
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
"""A collection of helper functions for interpolation."""

import numpy as np
import numpy.ma as ma


def extend_circular_coord(coord, points):
    """
    Return coordinates points with a shape extended by one
    This is common when dealing with circular coordinates.

    """
    modulus = np.array(coord.units.modulus or 0,
                       dtype=coord.dtype)
    points = np.append(points, points[0] + modulus)
    return points


def extend_circular_coord_and_data(coord, data, coord_dim):
    """
    Return coordinate points and a data array with a shape extended by one
    in the coord_dim axis. This is common when dealing with circular
    coordinates.

    """
    points = extend_circular_coord(coord, coord.points)
    data = extend_circular_data(data, coord_dim)
    return points, data


def extend_circular_data(data, coord_dim):
    coord_slice_in_cube = [slice(None)] * data.ndim
    coord_slice_in_cube[coord_dim] = slice(0, 1)

    # TODO: Restore this code after resolution of the following issue:
    # https://github.com/numpy/numpy/issues/478
    # data = np.append(cube.data,
    #                  cube.data[tuple(coord_slice_in_cube)],
    #                  axis=sample_dim)
    # This is the alternative, temporary workaround.
    # It doesn't use append on an nD mask.
    if not (isinstance(data, ma.MaskedArray) and
            not isinstance(data.mask, np.ndarray)) or \
            len(data.mask.shape) == 0:
        data = np.append(data,
                         data[tuple(coord_slice_in_cube)],
                         axis=coord_dim)
    else:
        new_data = np.append(data.data,
                             data.data[tuple(coord_slice_in_cube)],
                             axis=coord_dim)
        new_mask = np.append(data.mask,
                             data.mask[tuple(coord_slice_in_cube)],
                             axis=coord_dim)
        data = ma.array(new_data, mask=new_mask)
    return data


def get_xy_dim_coords(cube):
    """
    Return the x and y dimension coordinates from a cube.

    This function raises a ValueError if the cube does not contain one and
    only one set of x and y dimension coordinates. It also raises a ValueError
    if the identified x and y coordinates do not have coordinate systems that
    are equal.

    Args:

    * cube:
        An instance of :class:`iris.cube.Cube`.

    Returns:
        A tuple containing the cube's x and y dimension coordinates.

    """
    x_coords = cube.coords(axis='x', dim_coords=True)
    if len(x_coords) != 1:
        raise ValueError('Cube {!r} must contain a single 1D x '
                         'coordinate.'.format(cube.name()))
    x_coord = x_coords[0]

    y_coords = cube.coords(axis='y', dim_coords=True)
    if len(y_coords) != 1:
        raise ValueError('Cube {!r} must contain a single 1D y '
                         'coordinate.'.format(cube.name()))
    y_coord = y_coords[0]

    if x_coord.coord_system != y_coord.coord_system:
        raise ValueError("The cube's x ({!r}) and y ({!r}) "
                         "coordinates must have the same coordinate "
                         "system.".format(x_coord.name(), y_coord.name()))

    return x_coord, y_coord


def snapshot_grid(cube):
    """
    Helper function that returns deep copies of lateral dimension coordinates
    from a cube.

    """
    x, y = get_xy_dim_coords(cube)
    return x.copy(), y.copy()
