# (C) British Crown Copyright 2010 - 2016, Met Office
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
Defines a Trajectory class, and a routine to extract a sub-cube along a
trajectory.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import math

import numpy as np

import iris.coord_systems
import iris.coords
import iris.analysis
from iris.analysis._interpolate_private import \
    _nearest_neighbour_indices_ndcoords, linear as linear_regrid


class _Segment(object):
    """A single trajectory line segment: Two points, as described in the
    Trajectory class."""
    def __init__(self, p0, p1):
        # check keys
        if sorted(p0.keys()) != sorted(p1.keys()):
            raise ValueError("keys do not match")

        self.pts = [p0, p1]

        # calculate our length
        squares = 0
        for key in self.pts[0].keys():
            delta = self.pts[1][key] - self.pts[0][key]
            squares += delta * delta
        self.length = math.sqrt(squares)


class Trajectory(object):
    """A series of given waypoints with pre-calculated sample points."""

    def __init__(self, waypoints, sample_count=10):
        """
        Defines a trajectory using a sequence of waypoints.

        For example::

            waypoints = [{'latitude': 45, 'longitude': -60},
                         {'latitude': 45, 'longitude': 0}]
            Trajectory(waypoints)

        .. note:: All the waypoint dictionaries must contain the same
        coordinate names.

        Args:

        * waypoints
            A sequence of dictionaries, mapping coordinate names to values.

        Kwargs:

        * sample_count
            The number of sample positions to use along the trajectory.

        """
        self.waypoints = waypoints
        self.sample_count = sample_count

        # create line segments from the waypoints
        segments = [_Segment(self.waypoints[i], self.waypoints[i+1])
                    for i in range(len(self.waypoints) - 1)]

        # calculate our total length
        self.length = sum([seg.length for seg in segments])

        # generate our sampled points
        #: The trajectory points, as dictionaries of {coord_name: value}.
        self.sampled_points = []
        sample_step = self.length / (self.sample_count - 1)

        # start with the first segment
        cur_seg_i = 0
        cur_seg = segments[cur_seg_i]
        len_accum = cur_seg.length
        for p in range(self.sample_count):

            # calculate the sample position along our total length
            sample_at_len = p * sample_step

            # skip forward to the containing segment
            while len_accum < sample_at_len and cur_seg_i < len(segments):
                cur_seg_i += 1
                cur_seg = segments[cur_seg_i]
                len_accum += cur_seg.length

            # how far through the segment is our sample point?
            seg_start_len = len_accum - cur_seg.length
            seg_frac = (sample_at_len-seg_start_len) / cur_seg.length

            # sample each coordinate in this segment, to create a new
            # sampled point
            new_sampled_point = {}
            for key in cur_seg.pts[0].keys():
                seg_coord_delta = cur_seg.pts[1][key] - cur_seg.pts[0][key]
                new_sampled_point.update({key: cur_seg.pts[0][key] +
                                         seg_frac*seg_coord_delta})

            # add this new sampled point
            self.sampled_points.append(new_sampled_point)

    def __repr__(self):
        return 'Trajectory(%s, sample_count=%s)' % (self.waypoints,
                                                    self.sample_count)


def interpolate(cube, sample_points, method=None):
    """
    Extract a sub-cube at the given n-dimensional points.

    Args:

    * cube
        The source Cube.

    * sample_points
        A sequence of coordinate (name) - values pairs.

    Kwargs:

    * method
        Request "linear" interpolation (default) or "nearest" neighbour.
        Only nearest neighbour is available when specifying multi-dimensional
        coordinates.


    For example::

        sample_points = [('latitude', [45, 45, 45]),
        ('longitude', [-60, -50, -40])]
        interpolated_cube = interpolate(cube, sample_points)

    """
    if method not in [None, "linear", "nearest"]:
        raise ValueError("Unhandled interpolation specified : %s" % method)

    # Convert any coordinate names to coords
    points = []
    for coord, values in sample_points:
        if isinstance(coord, six.string_types):
            coord = cube.coord(coord)
        points.append((coord, values))
    sample_points = points

    # Do all value sequences have the same number of values?
    coord, values = sample_points[0]
    trajectory_size = len(values)
    for coord, values in sample_points[1:]:
        if len(values) != trajectory_size:
            raise ValueError('Lengths of coordinate values are inconsistent.')

    # Which dimensions are we squishing into the last dimension?
    squish_my_dims = set()
    for coord, values in sample_points:
        dims = cube.coord_dims(coord)
        for dim in dims:
            squish_my_dims.add(dim)

    # Derive the new cube's shape by filtering out all the dimensions we're
    # about to sample,
    # and then adding a new dimension to accommodate all the sample points.
    remaining = [(dim, size) for dim, size in enumerate(cube.shape) if dim
                 not in squish_my_dims]
    new_data_shape = [size for dim, size in remaining]
    new_data_shape.append(trajectory_size)

    # Start with empty data and then fill in the "column" of values for each
    # trajectory point.
    new_cube = iris.cube.Cube(np.empty(new_data_shape))
    new_cube.metadata = cube.metadata

    # Derive the mapping from the non-trajectory source dimensions to their
    # corresponding destination dimensions.
    remaining_dims = [dim for dim, size in remaining]
    dimension_remap = {dim: i for i, dim in enumerate(remaining_dims)}

    # Record a mapping from old coordinate IDs to new coordinates,
    # for subsequent use in creating updated aux_factories.
    coord_mapping = {}

    # Create all the non-squished coords
    for coord in cube.dim_coords:
        src_dims = cube.coord_dims(coord)
        if squish_my_dims.isdisjoint(src_dims):
            dest_dims = [dimension_remap[dim] for dim in src_dims]
            new_coord = coord.copy()
            new_cube.add_dim_coord(new_coord, dest_dims)
            coord_mapping[id(coord)] = new_coord
    for coord in cube.aux_coords:
        src_dims = cube.coord_dims(coord)
        if squish_my_dims.isdisjoint(src_dims):
            dest_dims = [dimension_remap[dim] for dim in src_dims]
            new_coord = coord.copy()
            new_cube.add_aux_coord(new_coord, dest_dims)
            coord_mapping[id(coord)] = new_coord

    # Create all the squished (non derived) coords, not filled in yet.
    trajectory_dim = len(remaining_dims)
    for coord in cube.dim_coords + cube.aux_coords:
        src_dims = cube.coord_dims(coord)
        if not squish_my_dims.isdisjoint(src_dims):
            points = np.array([coord.points.flatten()[0]] * trajectory_size)
            new_coord = iris.coords.AuxCoord(points,
                                             standard_name=coord.standard_name,
                                             long_name=coord.long_name,
                                             units=coord.units,
                                             bounds=None,
                                             attributes=coord.attributes,
                                             coord_system=coord.coord_system)
            new_cube.add_aux_coord(new_coord, trajectory_dim)
            coord_mapping[id(coord)] = new_coord

    for factory in cube.aux_factories:
        new_cube.add_aux_factory(factory.updated(coord_mapping))

    # Are the given coords all 1-dimensional? (can we do linear interp?)
    for coord, values in sample_points:
        if coord.ndim > 1:
            if method == "linear":
                msg = "Cannot currently perform linear interpolation for " \
                      "multi-dimensional coordinates."
                raise iris.exceptions.CoordinateMultiDimError(msg)
            method = "nearest"
            break

    if method in ["linear", None]:
        for i in range(trajectory_size):
            point = [(coord, values[i]) for coord, values in sample_points]
            column = linear_regrid(cube, point)
            new_cube.data[..., i] = column.data
            # Fill in the empty squashed (non derived) coords.
            for column_coord in column.dim_coords + column.aux_coords:
                src_dims = cube.coord_dims(column_coord)
                if not squish_my_dims.isdisjoint(src_dims):
                    if len(column_coord.points) != 1:
                        raise Exception("Expected to find exactly one point. Found %d" % len(column_coord.points))
                    new_cube.coord(column_coord.name()).points[i] = column_coord.points[0]

    elif method == "nearest":
        # Use a cache with _nearest_neighbour_indices_ndcoords()
        cache = {}
        for i in range(trajectory_size):
            point = [(coord, values[i]) for coord, values in sample_points]
            column_index = _nearest_neighbour_indices_ndcoords(cube, point, cache=cache)
            column = cube[column_index]
            new_cube.data[..., i] = column.data
            # Fill in the empty squashed (non derived) coords.
            for column_coord in column.dim_coords + column.aux_coords:
                src_dims = cube.coord_dims(column_coord)
                if not squish_my_dims.isdisjoint(src_dims):
                    if len(column_coord.points) != 1:
                        raise Exception("Expected to find exactly one point. Found %d" % len(column_coord.points))
                    new_cube.coord(column_coord.name()).points[i] = column_coord.points[0]

    return new_cube


class UnstructuredNearestNeigbourRegridder(object):
    """
    Encapsulate the operation of :meth:`iris.analysis.trajectory.interpolate`
    with given source and target grids.

    TODO: cache the necessary bits of the operation so re-use can actually
    be more efficient.

    """
    def __init__(self, src_cube, target_grid):
        """
        A nearest-neighbour regridder to perform regridding from the source
        grid to the target grid.

        This can then be applied to any source data with the same structure as
        the original 'src_cube'.

        Args:

        * src_cube:
            The :class:`~iris.cube.Cube` defining the source grid.
            The X and Y coordinates must be mapped over the same dimensions.

        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.
            It must have only 2 dimensions.
            The X and Y coordinates must be one-dimensional and mapped to
            different dimensions.

        Returns:
            regridder : (object)

            A callable object with the interface:
                `result_cube = regridder(data)`

            where `data` is a cube with the same grid as the original
            `src_cube`, that is to be regridded to the `target_grid`.

        """
        # Store the essential stuff
        self.src_cube = src_cube
        self.grid_cube = target_grid

        # Quickly check the source data structure.
        # TODO: replace asserts with code to raise user-intelligible errors.

        # Has unique X and Y coords.
        x_co = src_cube.coord(axis='x')
        y_co = src_cube.coord(axis='y')
        # They have a single common dimension, WHICH IS THE LAST.
        src_ndim = src_cube.ndim
        assert src_cube.coord_dims(x_co) == (src_ndim - 1,)
        assert src_cube.coord_dims(y_co) == (src_ndim - 1,)

        # Quickly check the target grid structure.
        # TODO: ensure any errors are intelligible to the user.
        # Has only 2 dims.
        assert target_grid.ndim == 2
        # Has unique X and Y coords.
        x_co = target_grid.coord(axis='x')
        y_co = target_grid.coord(axis='y')
        # Each has a dimension to itself.
        x_dims = target_grid.coord_dims(x_co)
        y_dims = target_grid.coord_dims(y_co)
        assert len(x_dims) == 1
        assert len(y_dims) == 1
        assert x_dims != y_dims

        # Pre-calculate the sample points that will be needed.
        # These are cast as a 'trajectory' to suit the method used.
        x_vals = target_grid.coord('longitude').points
        y_vals = target_grid.coord('latitude').points
        x_2d, y_2d = np.meshgrid(x_vals, y_vals)
        self.trajectory = (('longitude', x_2d.flatten()),
                           ('latitude', y_2d.flatten()))

    def __call__(self, src_cube):
        # Check source cube matches original.
        # For now, just a shape match will do.
        # TODO: implement a more intelligent equivalence check.
        # TODO: replace asserts with code to raise user-intelligible errors.
        assert src_cube.shape == self.src_cube.shape

        # Get the basic interpolated results.
        result_trajectory_cube = interpolate(src_cube, self.trajectory,
                                             method='nearest')

        # Reconstruct this as a cube "like" the source data.
        # TODO: sort out aux-coords, cell methods, cell measures ??

        # The shape is that of source data, minus the last dim, plus the target
        # grid dimensions.
        target_shape = (list(src_cube.shape)[:-1] + list(self.grid_cube.shape))
        data_2d_x_and_y = result_trajectory_cube.data.reshape(target_shape)

        # Make a new result cube with the reshaped data.
        result_cube = iris.cube.Cube(data_2d_x_and_y)
        result_cube.metadata = src_cube.metadata

        # Copy the 'preceding' dim coords from the source cube.
        n_other_dims = src_cube.ndim - 1
        for i_dim in range(n_other_dims):
            co = src_cube.coord(dimensions=(i_dim,), dim_coords=True)
            result_cube.add_dim_coord(co.copy(), i_dim)

        # Copy the 'trailing' lat+lon coords from the grid cube.
        for i_dim in (0, 1):
            co = self.grid_cube.coord(dimensions=(i_dim,))
            result_cube.add_dim_coord(co.copy(), i_dim + n_other_dims)

        return result_cube
