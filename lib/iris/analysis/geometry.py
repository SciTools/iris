# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Various utilities related to geometric operations.

.. note::
    This module requires :mod:`shapely`.

"""

import warnings

import numpy as np
from shapely.geometry import Polygon

import iris.exceptions


def _extract_relevant_cube_slice(cube, geometry):
    """Calculate geometry intersection with spatial region defined by cube.

    This helper method returns the tuple
    (subcube, x_coord_of_subcube, y_coord_of_subcube,
    (min_x_index, min_y_index, max_x_index, max_y_index)).

    If cube and geometry don't overlap, returns None.

    """
    # Validate the input parameters
    if not cube.coords(axis="x") or not cube.coords(axis="y"):
        raise ValueError("The cube must contain x and y axes.")

    x_coords = cube.coords(axis="x")
    y_coords = cube.coords(axis="y")
    if len(x_coords) != 1 or len(y_coords) != 1:
        raise ValueError(
            "The cube must contain one, and only one, coordinate "
            "for each of the x and y axes."
        )

    x_coord = x_coords[0]
    y_coord = y_coords[0]
    if not (x_coord.has_bounds() and y_coord.has_bounds()):
        raise ValueError("Both horizontal coordinates must have bounds.")

    if x_coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(x_coord)
    if y_coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError(y_coord)

    # bounds of cube dimensions
    x_bounds = x_coord.bounds
    y_bounds = y_coord.bounds

    # identify ascending/descending coordinate dimensions
    x_ascending = x_coord.points[1] - x_coord.points[0] > 0.0
    y_ascending = y_coord.points[1] - y_coord.points[0] > 0.0

    # identify upper/lower bounds of coordinate dimensions
    x_bounds_lower = x_bounds[:, 0] if x_ascending else x_bounds[:, 1]
    y_bounds_lower = y_bounds[:, 0] if y_ascending else y_bounds[:, 1]
    x_bounds_upper = x_bounds[:, 1] if x_ascending else x_bounds[:, 0]
    y_bounds_upper = y_bounds[:, 1] if y_ascending else y_bounds[:, 0]

    # find indices of coordinate bounds to fully cover geometry
    x_min_geom, y_min_geom, x_max_geom, y_max_geom = geometry.bounds
    try:
        x_min_ix = np.where(x_bounds_lower <= x_min_geom)[0]
        x_min_ix = x_min_ix[np.argmax(x_bounds_lower[x_min_ix])]
    except ValueError:
        warnings.warn(
            "The geometry exceeds the cube's x dimension at the lower end.",
            category=iris.exceptions.IrisGeometryExceedWarning,
        )
        x_min_ix = 0 if x_ascending else x_coord.points.size - 1

    try:
        x_max_ix = np.where(x_bounds_upper >= x_max_geom)[0]
        x_max_ix = x_max_ix[np.argmin(x_bounds_upper[x_max_ix])]
    except ValueError:
        warnings.warn(
            "The geometry exceeds the cube's x dimension at the upper end.",
            category=iris.exceptions.IrisGeometryExceedWarning,
        )
        x_max_ix = x_coord.points.size - 1 if x_ascending else 0

    try:
        y_min_ix = np.where(y_bounds_lower <= y_min_geom)[0]
        y_min_ix = y_min_ix[np.argmax(y_bounds_lower[y_min_ix])]
    except ValueError:
        warnings.warn(
            "The geometry exceeds the cube's y dimension at the lower end.",
            category=iris.exceptions.IrisGeometryExceedWarning,
        )
        y_min_ix = 0 if y_ascending else y_coord.points.size - 1

    try:
        y_max_ix = np.where(y_bounds_upper >= y_max_geom)[0]
        y_max_ix = y_max_ix[np.argmin(y_bounds_upper[y_max_ix])]
    except ValueError:
        warnings.warn(
            "The geometry exceeds the cube's y dimension at the upper end.",
            category=iris.exceptions.IrisGeometryExceedWarning,
        )
        y_max_ix = y_coord.points.size - 1 if y_ascending else 0

    # extract coordinate values at these indices
    x_min = x_bounds_lower[x_min_ix]
    x_max = x_bounds_upper[x_max_ix]
    y_min = y_bounds_lower[y_min_ix]
    y_max = y_bounds_upper[y_max_ix]

    # switch min and max if necessary, to create slice objects later on
    if x_min_ix > x_max_ix:
        x_min_ix, x_max_ix = x_max_ix, x_min_ix
    if y_min_ix > y_max_ix:
        y_min_ix, y_max_ix = y_max_ix, y_min_ix
    bnds_ix = x_min_ix, y_min_ix, x_max_ix, y_max_ix

    # cut the relevant part from the original cube
    coord_constr = {
        x_coord.name(): lambda x: x_min <= x.point <= x_max,
        y_coord.name(): lambda y: y_min <= y.point <= y_max,
    }
    constraint = iris.Constraint(coord_values=coord_constr)
    subcube = cube.extract(constraint)

    if subcube is None:
        return None
    else:
        x_coord = subcube.coord(axis="x")
        y_coord = subcube.coord(axis="y")
        return subcube, x_coord, y_coord, bnds_ix


def geometry_area_weights(cube, geometry, normalize=False):
    """Return the array of weights corresponding to the area of overlap.

    Return the array of weights corresponding to the area of overlap between
    the cells of cube's horizontal grid, and the given shapely geometry.

    The returned array is suitable for use with :const:`iris.analysis.MEAN`.

    The cube must have bounded horizontal coordinates.

    .. note::
        This routine works in Euclidean space. Area calculations do not
        account for the curvature of the Earth. And care must be taken to
        ensure any longitude values are expressed over a suitable interval.

    .. note::
        This routine currently does not handle all out-of-bounds cases
        correctly. In cases where both the coordinate bounds and the
        geometry's bounds lie outside the physically realistic range
        (i.e., abs(latitude) > 90., as it is commonly the case when
        bounds are constructed via guess_bounds()), the weights
        calculation might be wrong. In this case, a UserWarning will
        be issued.

    .. note::

        This function does not maintain laziness when called; it realises data.
        See more at :doc:`/userguide/real_and_lazy_data`.

    Args:

    * cube (:class:`iris.cube.Cube`):
        A Cube containing a bounded, horizontal grid definition.
    * geometry (a shapely geometry instance):
        The geometry of interest. To produce meaningful results this geometry
        must have a non-zero area. Typically a Polygon or MultiPolygon.

    Kwargs:

    * normalize:
        Calculate each individual cell weight as the cell area overlap between
        the cell and the given shapely geometry divided by the total cell area.
        Default is False.

    """
    # extract smallest subcube containing geometry
    shape = cube.shape
    extraction_results = _extract_relevant_cube_slice(cube, geometry)

    # test if there is overlap between cube and geometry
    if extraction_results is None:
        return np.zeros(shape)

    subcube, subx_coord, suby_coord, bnds_ix = extraction_results
    x_min_ix, y_min_ix, x_max_ix, y_max_ix = bnds_ix

    # prepare the weights array
    subshape = list(cube.shape)
    x_dim = cube.coord_dims(subx_coord)[0]
    y_dim = cube.coord_dims(suby_coord)[0]
    subshape[x_dim] = subx_coord.shape[0]
    subshape[y_dim] = suby_coord.shape[0]
    subx_bounds = subx_coord.bounds
    suby_bounds = suby_coord.bounds
    subweights = np.empty(subshape, np.float32)

    # calculate the area weights
    for nd_index in np.ndindex(subweights.shape):
        xi = nd_index[x_dim]
        yi = nd_index[y_dim]
        x0, x1 = subx_bounds[xi]
        y0, y1 = suby_bounds[yi]
        polygon = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        subweights[nd_index] = polygon.intersection(geometry).area
        if normalize:
            subweights[nd_index] /= polygon.area

    # pad the calculated weights with zeros to match original cube shape
    weights = np.zeros(shape, np.float32)
    slices = []
    for i in range(weights.ndim):
        if i == x_dim:
            slices.append(slice(x_min_ix, x_max_ix + 1))
        elif i == y_dim:
            slices.append(slice(y_min_ix, y_max_ix + 1))
        else:
            slices.append(slice(None))

    weights[tuple(slices)] = subweights

    # Fix for the limitation of iris.analysis.MEAN weights handling.
    # Broadcast the array to the full shape of the cube
    weights = np.broadcast_arrays(weights, cube.data)[0]

    return weights
