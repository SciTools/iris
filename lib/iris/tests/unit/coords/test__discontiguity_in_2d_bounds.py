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
"""Unit tests for the `iris.coords._discontiguity_in_2d_bounds` function."""


from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import _discontiguity_in_2d_bounds


class Test(tests.IrisTest):
    def setUp(self):
        self.lon_bounds_3by3 = np.array(
            [[[0, 1, 1, 0], [1, 2, 2, 1], [2, 3, 3, 2]],
             [[0, 1, 1, 0], [1, 2, 2, 1], [2, 3, 3, 2]],
             [[0, 1, 1, 0], [1, 2, 2, 1], [2, 3, 3, 2]]])
        self.lat_bounds_3by3 = np.array(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
             [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
             [[2, 2, 3, 3], [2, 2, 3, 3], [2, 2, 3, 3]]])
        self.discontiguous_bds = np.array(
            [[[0, 1, 1, 0], [2, 3, 3, 2]],
             [[1, 2, 2, 1], [2, 3, 3, 2]]])

    def test_bds_wrong_shape_not_3d(self):
        bds = self.lon_bounds_3by3[0]
        err_msg = 'Bounds for 2D coordinates must be 3-dimensional and have ' \
                  '4 bounds per point'
        with self.assertRaisesRegexp(ValueError, err_msg):
            _discontiguity_in_2d_bounds(bds)

    def test_bds_wrong_shape_not_4_bounds(self):
        bds = self.lon_bounds_3by3[:, :, :-2]
        err_msg = 'Bounds for 2D coordinates must be 3-dimensional and have ' \
                  '4 bounds per point'
        with self.assertRaisesRegexp(ValueError, err_msg):
            _discontiguity_in_2d_bounds(bds)

    def test_contiguous_both_dirs(self):
        bds = self.lon_bounds_3by3
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertTrue(all_eq)
        self.assertTrue(not diffs_along_x.any())
        self.assertTrue(not diffs_along_y.any())

    def test_discontiguous_along_x(self):
        bds = self.lon_bounds_3by3[:, ::2, :]
        exp_result = np.array([1, 1, 1]).reshape(3, 1)
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertFalse(all_eq)
        self.assertTrue(not diffs_along_y.any())
        self.assertArrayEqual(diffs_along_x, exp_result)

    def test_discontiguous_along_y(self):
        bds = self.lat_bounds_3by3[::2, :, :]
        exp_result = np.array([1, 1, 1]).reshape(1, 3)
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertFalse(all_eq)
        self.assertTrue(not diffs_along_x.any())
        self.assertArrayEqual(diffs_along_y, exp_result)

    def test_discontiguous_along_x_and_y(self):
        bds = self.discontiguous_bds
        x_exp_result = np.array([1, 0]).reshape(2, 1)
        y_exp_result = np.array([1, 0]).reshape(1, 2)
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertFalse(all_eq)
        self.assertArrayEqual(diffs_along_x, x_exp_result)
        self.assertArrayEqual(diffs_along_y, y_exp_result)

    def test_bds_shape_1_by_1(self):
        bds = self.lon_bounds_3by3[:1, :1, :]
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertTrue(all_eq)
        self.assertIsNone(diffs_along_x)
        self.assertIsNone(diffs_along_y)

    def test_bds_shape_1_by_2_contiguous_along_x(self):
        bds = self.lon_bounds_3by3[:1, :, :]
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertTrue(all_eq)
        self.assertIsNone(diffs_along_x)
        self.assertTrue(not diffs_along_y.any())

    def test_bds_shape_2_by_1_contiguous_along_y(self):
        bds = self.lon_bounds_3by3[:, :1, :]
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertTrue(all_eq)
        self.assertIsNone(diffs_along_y)
        self.assertTrue(not diffs_along_x.any())

    def test_contiguous_abs_tol(self):
        bds = self.discontiguous_bds / 1000
        x_exp_result = np.array([1, 0]).reshape(2, 1) / 1000
        y_exp_result = np.array([1, 0]).reshape(1, 2) / 1000
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(
            bds, atol=1e-2)
        self.assertTrue(all_eq)
        self.assertArrayAlmostEqual(diffs_along_x, x_exp_result)
        self.assertArrayAlmostEqual(diffs_along_y, y_exp_result)

    def test_discontiguous_abs_tol(self):
        bds = self.discontiguous_bds
        x_exp_result = np.array([1, 0]).reshape(2, 1)
        y_exp_result = np.array([1, 0]).reshape(1, 2)
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(
            bds, atol=1e-2)
        self.assertFalse(all_eq)
        self.assertArrayEqual(diffs_along_x, x_exp_result)
        self.assertArrayEqual(diffs_along_y, y_exp_result)

    def test_contiguous_mod_360(self):
        bds = np.array([[[170, 180, 180, 170], [-180, -170, -170, -180]],
                        [[170, 180, 180, 170], [-180, -170, -170, -180]]])
        all_eq, diffs_along_x, diffs_along_y = _discontiguity_in_2d_bounds(bds)
        self.assertTrue(all_eq)
        self.assertTrue(not diffs_along_x.any())
        self.assertTrue(not diffs_along_y.any())


if __name__ == "__main__":
    tests.main()
