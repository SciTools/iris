# (C) British Crown Copyright 2016, Met Office
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
Unit tests for :class:`iris.analysis.trajectory.Trajectory`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import numpy as np

from iris.analysis.trajectory import Trajectory
from iris.tests.stock import simple_3d


class Test___init__(tests.IrisTest):
    def test_2_points(self):
        # basic 2-seg line along x
        waypoints = [{'lat': 0, 'lon': 0}, {'lat': 1, 'lon': 2}]
        trajectory = Trajectory(waypoints, sample_count=5)

        self.assertEqual(trajectory.length, np.sqrt(5))
        self.assertEqual(trajectory.sample_count, 5)
        self.assertEqual(trajectory.sampled_points,
                         [{'lat': 0.0, 'lon': 0.0},
                          {'lat': 0.25, 'lon': 0.5},
                          {'lat': 0.5, 'lon': 1.0},
                          {'lat': 0.75, 'lon': 1.5},
                          {'lat': 1.0, 'lon': 2.0}])

    def test_3_points(self):
        # basic 2-seg line along x
        waypoints = [{'lat': 0, 'lon': 0}, {'lat': 0, 'lon': 1},
                     {'lat': 0, 'lon': 2}]
        trajectory = Trajectory(waypoints, sample_count=21)

        self.assertEqual(trajectory.length, 2.0)
        self.assertEqual(trajectory.sample_count, 21)
        self.assertEqual(trajectory.sampled_points[19],
                         {'lat': 0.0, 'lon': 1.9000000000000001})

    def test_zigzag(self):
        # 4-seg m-shape
        waypoints = [{'lat': 0, 'lon': 0}, {'lat': 1, 'lon': 1},
                     {'lat': 0, 'lon': 2}, {'lat': 1, 'lon': 3},
                     {'lat': 0, 'lon': 4}]
        trajectory = Trajectory(waypoints, sample_count=33)

        self.assertEqual(trajectory.length, 5.6568542494923806)
        self.assertEqual(trajectory.sample_count, 33)
        self.assertEqual(trajectory.sampled_points[31],
                         {'lat': 0.12499999999999989, 'lon': 3.875})


class Test__get_interp_points(tests.IrisTest):
    def test_basic(self):
        waypoints = [{'lat': 0}, {'lat': 1}]
        sample_count = 5
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory._get_interp_points()
        expected_points = list(np.linspace(0, 1, sample_count))

        self.assertEqual(len(result), len(waypoints[0]))
        self.assertEqual(len(result[0][1]), sample_count)
        self.assertEqual(result[0][1], expected_points)

    def test_2d(self):
        waypoints = [{'lat': 0, 'lon': 0}, {'lat': 1, 'lon': 2}]
        sample_count = 5
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory._get_interp_points()

        self.assertEqual(len(result), len(waypoints[0]))
        self.assertEqual(len(result[0][1]), sample_count)
        self.assertEqual(len(result[1][1]), sample_count)
        self.assertEqual(result[0][0], 'lat')
        self.assertEqual(result[1][0], 'lon')

    def test_3d(self):
        waypoints = [{'y': 0, 'x': 0, 'z': 2}, {'y': 1, 'x': 2, 'z': 10}]
        sample_count = 5
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory._get_interp_points()

        self.assertEqual(len(result), len(waypoints[0]))
        self.assertEqual(len(result[0][1]), sample_count)
        self.assertEqual(len(result[1][1]), sample_count)
        self.assertEqual(len(result[2][1]), sample_count)
        self.assertEqual(result[0][0], 'y')
        self.assertEqual(result[1][0], 'x')
        self.assertEqual(result[2][0], 'z')


class Test_interpolate(tests.IrisTest):
    def test_cube__simple_3d(self):
        # Test that an 'index' coord is added to the resultant cube.
        cube = simple_3d()
        waypoints = [{'latitude': 40, 'longitude': 40},
                     {'latitude': 0, 'longitude': 0}]
        sample_count = 3
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory.interpolate(cube)
        coord_names = [c.name() for c in result.coords(dim_coords=True)]
        new_coord = result.coord(coord_names[1])

        self.assertEqual(result.ndim, cube.ndim - 1)
        self.assertEqual(coord_names[1], 'index')
        self.assertEqual(len(new_coord.points), sample_count)

    def test_call(self):
        # Test that :func:`iris.analysis.trajectory.interpolate` is called by
        # `Trajectory.interpolate`.
        cube = simple_3d()
        to_patch = 'iris.analysis.trajectory.interpolate'
        waypoints = [{'latitude': 40, 'longitude': 40},
                     {'latitude': 0, 'longitude': 0}]
        sample_count = 3
        trajectory = Trajectory(waypoints, sample_count=sample_count)

        with mock.patch(to_patch, return_value=cube) as mock_interpolate:
            trajectory.interpolate(cube)
        mock_interpolate.assert_called_once()


if __name__ == "__main__":
    tests.main()
