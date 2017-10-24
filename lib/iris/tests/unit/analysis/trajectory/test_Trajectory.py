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

import numpy as np

from iris.analysis.trajectory import Trajectory


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


if __name__ == "__main__":
    tests.main()
