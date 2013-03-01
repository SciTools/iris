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
Defines a Trajectory class, and a routine to extract a sub-cube along a trajectory.

"""

import math

import numpy as np

import iris.coord_systems
import iris.coords
import iris.analysis


class _Segment(object):
    """A single trajectory line segment: Two points, as described in the Trajectory class."""
    def __init__(self, p0, p1):
        #check keys
        if sorted(p0.keys()) != sorted(p1.keys()):
            raise ValueError("keys do not match")
        
        self.pts = [p0, p1]

        #calculate our length
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
        
            waypoints = [{'latitude': 45, 'longitude': -60}, {'latitude': 45, 'longitude': 0}]
            Trajectory(waypoints)
        
        .. note:: All the waypoint dictionaries must contain the same coordinate names.
        
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
        segments = [_Segment(self.waypoints[i], self.waypoints[i+1]) for i in range(len(self.waypoints) - 1)]

        # calculate our total length
        self.length = sum([seg.length for seg in segments])

        # generate our sampled points
        self.sampled_points = []
        sample_step = self.length / (self.sample_count - 1)
        
        #start with the first segment
        cur_seg_i = 0
        cur_seg = segments[cur_seg_i]
        len_accum = cur_seg.length
        for p in range(self.sample_count):

            # calculate the sample position along our total length
            sample_at_len = p * sample_step
            
            # skip forward to the containing segment
            while(len_accum < sample_at_len and cur_seg_i < len(segments)):
                cur_seg_i += 1
                cur_seg = segments[cur_seg_i]
                len_accum += cur_seg.length
            
            # how far through the segment is our sample point?
            seg_start_len = len_accum - cur_seg.length
            seg_frac = (sample_at_len-seg_start_len) / cur_seg.length

            # sample each coordinate in this segment, to create a new sampled point
            new_sampled_point = {}
            for key in cur_seg.pts[0].keys():
                seg_coord_delta = cur_seg.pts[1][key] - cur_seg.pts[0][key]
                new_sampled_point.update({key: cur_seg.pts[0][key] + seg_frac*seg_coord_delta})
        
            # add this new sampled point
            self.sampled_points.append(new_sampled_point)
            
    def __repr__(self):
        return 'Trajectory(%s, sample_count=%s)' % (self.waypoints, self.sample_count)


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
        Only nearest neighbour is available when specifying multi-dimensional coordinates.
        

    For example::
            sample_points = [('latitude', [45, 45, 45]), ('longitude', [-60, -50, -40])]
            interpolated_cube = interpolate(cube, sample_points)

    """
    if method not in [None, "linear", "nearest"]:
        raise ValueError("Unhandled interpolation specified : %s" % method)
    
    # Convert any coordinate names to coords
    points = []
    for coord, values in sample_points:
        if isinstance(coord, basestring):
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


    # Derive the new cube's shape by filtering out all the dimensions we're about to sample,
    # and then adding a new dimension to accommodate all the sample points.
    remaining = [(dim, size) for dim, size in enumerate(cube.shape) if dim not in squish_my_dims]
    new_data_shape = [size for dim, size in remaining]
    new_data_shape.append(trajectory_size)

    # Start with empty data and then fill in the "column" of values for each trajectory point.
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
                raise iris.exceptions.CoordinateMultiDimError("Cannot currently perform linear interpolation for multi-dimensional coordinates.")
            method = "nearest" 
            break

    # Use a cache with _nearest_neighbour_indices_ndcoords()
    cache = {}
    
    for i in range(trajectory_size):
        point = [(coord, values[i]) for coord, values in sample_points]
        
        if method in ["linear", None]:
            column = iris.analysis.interpolate.linear(cube, point)
            new_cube.data[..., i] = column.data
        elif method == "nearest":
            column_index = iris.analysis.interpolate._nearest_neighbour_indices_ndcoords(cube, point, cache=cache)
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

