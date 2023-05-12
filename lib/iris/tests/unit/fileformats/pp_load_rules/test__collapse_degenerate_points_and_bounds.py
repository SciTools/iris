# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:func:`iris.fileformats.pp_load_rules._collapse_degenerate_points_and_bounds`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.fileformats.pp_load_rules import (
    _collapse_degenerate_points_and_bounds,
)


class Test(tests.IrisTest):
    def test_scalar(self):
        array = np.array(1)
        points, bounds = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(points, array)
        self.assertIsNone(bounds)

    def test_1d_nochange(self):
        array = np.array([1, 1, 3])
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(result, array)

    def test_1d_collapse(self):
        array = np.array([1, 1, 1])
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(result, np.array([1]))

    def test_2d_nochange(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(result, array)

    def test_2d_collapse_dim0(self):
        array = np.array([[1, 2, 3], [1, 2, 3]])
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(result, np.array([[1, 2, 3]]))

    def test_2d_collapse_dim1(self):
        array = np.array([[1, 1, 1], [2, 2, 2]])
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(result, np.array([[1], [2]]))

    def test_2d_collapse_both(self):
        array = np.array([[3, 3, 3], [3, 3, 3]])
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(result, np.array([[3]]))

    def test_3d(self):
        array = np.array([[[3, 3, 3], [4, 4, 4]], [[3, 3, 3], [4, 4, 4]]])
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertArrayEqual(result, np.array([[[3], [4]]]))

    def test_multiple_odd_dims(self):
        # Test to ensure multiple collapsed dimensions don't interfere.
        # make a 5-D array where dimensions 0, 2 and 3 are degenerate.
        array = np.arange(3**5).reshape([3] * 5)
        array[1:] = array[0:1]
        array[:, :, 1:] = array[:, :, 0:1]
        array[:, :, :, 1:] = array[:, :, :, 0:1]
        result, _ = _collapse_degenerate_points_and_bounds(array)
        self.assertEqual(array.shape, (3, 3, 3, 3, 3))
        self.assertEqual(result.shape, (1, 3, 1, 1, 3))
        self.assertTrue(np.all(result == array[0:1, :, 0:1, 0:1, :]))

    def test_bounds_collapse(self):
        points = np.array([1, 1, 1])
        bounds = np.array([[0, 1], [0, 1], [0, 1]])
        result_pts, result_bds = _collapse_degenerate_points_and_bounds(
            points, bounds
        )
        self.assertArrayEqual(result_pts, np.array([1]))
        self.assertArrayEqual(result_bds, np.array([[0, 1]]))

    def test_bounds_no_collapse(self):
        points = np.array([1, 1, 1])
        bounds = np.array([[0, 1], [0, 1], [0, 2]])
        result_pts, result_bds = _collapse_degenerate_points_and_bounds(
            points, bounds
        )
        self.assertArrayEqual(result_pts, points)
        self.assertArrayEqual(result_bds, bounds)


if __name__ == "__main__":
    tests.main()
