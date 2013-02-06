# (C) British Crown Copyright 2010 - 2012, Met Office
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
Various utilities related to geometric operations.

.. note::
    This module requires :mod:`shapely`.

"""

from shapely.geometry import Polygon

import numpy as np

import iris.exceptions


def geometry_area_weights(cube, geometry):
    """
    Returns the array of weights corresponding to the area of overlap between
    the cells of cube's horizontal grid, and the given shapely geometry.
    
    The returned array is suitable for use with :const:`iris.analysis.MEAN`.
    
    The cube must have bounded horizontal coordinates.
    
    .. note::
        This routine works in Euclidean space. Area calculations do not
        account for the curvature of the Earth. And care must be taken to
        ensure any longitude values are expressed over a suitable interval.
    
    Args:
    
    * cube (:class:`iris.cube.Cube`):
        A Cube containing a bounded, horizontal grid definition.
    * geometry (a shapely geometry instance):
        The geometry of interest. To produce meaningful results this geometry
        must have a non-zero area. Typically a Polygon or MultiPolygon.

    """
    # Validate the input parameters
    if not cube.coords(axis='x') or not cube.coords(axis='y'):
        raise ValueError('The cube must contain x and y axes.')

    x_coords = cube.coords(axis='x')
    y_coords = cube.coords(axis='y')
    if len(x_coords) != 1 or len(y_coords) != 1:
        raise ValueError('The cube must contain one, and only one, coordinate for each of the x and y axes.')

    x_coord = x_coords[0]
    y_coord = y_coords[0]
    if not (x_coord.has_bounds() and y_coord.has_bounds()):
        raise ValueError('Both horizontal coordinates must have bounds.')

    if x_coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(x_coord)
    if y_coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(y_coord)

    # Figure out the shape of the horizontal dimensions
    shape = [1] * len(cube.shape)
    x_dim = cube.coord_dims(x_coord)[0]
    y_dim = cube.coord_dims(y_coord)[0]
    shape[x_dim] = x_coord.shape[0]
    shape[y_dim] = y_coord.shape[0]
    weights = np.empty(shape, np.float32)

    # Calculate the area weights
    x_bounds = x_coord.bounds
    y_bounds = y_coord.bounds
    for nd_index in np.ndindex(weights.shape):
        xi = nd_index[x_dim]
        yi = nd_index[y_dim]
        x0, x1 = x_bounds[xi]
        y0, y1 = y_bounds[yi]
        polygon = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        weights[nd_index] = polygon.intersection(geometry).area

    # Fix for the limitation of iris.analysis.MEAN weights handling.
    # Broadcast the array to the full shape of the cube
    weights = np.broadcast_arrays(weights, cube.data)[0]

    return weights
