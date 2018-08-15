# (C) British Crown Copyright 2018, Met Office
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
"""Unit tests for the `iris.coords._get_2d_coord_bound_grid` function."""


from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import _get_2d_coord_bound_grid


class Test(tests.IrisTest):
    def setUp(self):
        self.lon_bounds = np.array(
            [[[0, 1, 1, 0], [1, 2, 2, 1]],
             [[0, 1, 1, 0], [1, 2, 2, 1]]])
        self.lat_bounds = np.array(
            [[[0, 0, 1, 1], [0, 0, 1, 1]],
             [[1, 1, 2, 2], [1, 1, 2, 2]]])

    def test_bds_wrong_shape_not_3d(self):
        bds = self.lon_bounds[0]
        err_msg = 'Bounds for 2D coordinates must be 3-dimensional and have ' \
                  '4 bounds per point'
        with self.assertRaisesRegexp(ValueError, err_msg):
            _get_2d_coord_bound_grid(bds)

    def test_bds_wrong_shape_not_4_bounds(self):
        bds = self.lon_bounds[:, :, :-2]
        err_msg = 'Bounds for 2D coordinates must be 3-dimensional and have ' \
                  '4 bounds per point'
        with self.assertRaisesRegexp(ValueError, err_msg):
            _get_2d_coord_bound_grid(bds)

    def test_lon_bounds(self):
        bds = self.lon_bounds
        expected = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        result = _get_2d_coord_bound_grid(bds)
        self.assertArrayEqual(result, expected)

    def test_lat_bounds(self):
        bds = self.lat_bounds
        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        result = _get_2d_coord_bound_grid(bds)
        self.assertArrayEqual(result, expected)


if __name__ == "__main__":
    tests.main()
