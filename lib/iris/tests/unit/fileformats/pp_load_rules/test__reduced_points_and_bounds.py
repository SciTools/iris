# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for
:func:`iris.fileformats.pp_load_rules._reduce_points_and_bounds`.

"""

import numpy as np

from iris.fileformats.pp_load_rules import _reduce_points_and_bounds
from iris.tests._shared_utils import assert_array_equal


class Test:
    def test_scalar(self):
        array = np.array(1)
        dims, result, bounds = _reduce_points_and_bounds(array)
        assert_array_equal(result, array)
        assert dims is None
        assert bounds is None

    def test_1d_nochange(self):
        array = np.array([1, 2, 3])
        dims, result, _ = _reduce_points_and_bounds(array)
        assert_array_equal(result, array)
        assert dims == (0,)

    def test_1d_collapse(self):
        array = np.array([1, 1, 1])
        dims, result, _ = _reduce_points_and_bounds(array)
        assert_array_equal(result, np.array(1))
        assert dims is None

    def test_2d_nochange(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        dims, result, _ = _reduce_points_and_bounds(array)
        assert_array_equal(result, array)
        assert dims == (0, 1)

    def test_2d_collapse_dim0(self):
        array = np.array([[1, 2, 3], [1, 2, 3]])
        dims, result, _ = _reduce_points_and_bounds(array)
        assert_array_equal(result, np.array([1, 2, 3]))
        assert dims == (1,)

    def test_2d_collapse_dim1(self):
        array = np.array([[1, 1, 1], [2, 2, 2]])
        dims, result, _ = _reduce_points_and_bounds(array)
        assert_array_equal(result, np.array([1, 2]))
        assert dims == (0,)

    def test_2d_collapse_both(self):
        array = np.array([[3, 3, 3], [3, 3, 3]])
        dims, result, _ = _reduce_points_and_bounds(array)
        assert_array_equal(result, np.array(3))
        assert dims is None

    def test_3d(self):
        array = np.array([[[3, 3, 3], [4, 4, 4]], [[3, 3, 3], [4, 4, 4]]])
        dims, result, _ = _reduce_points_and_bounds(array)
        assert_array_equal(result, np.array([3, 4]))
        assert dims == (1,)

    def test_bounds_collapse(self):
        points = np.array([1, 1, 1])
        bounds = np.array([[0, 2], [0, 2], [0, 2]])
        result_dims, result_pts, result_bds = _reduce_points_and_bounds(
            points, (bounds[..., 0], bounds[..., 1])
        )
        assert_array_equal(result_pts, np.array(1))
        assert_array_equal(result_bds, np.array([0, 2]))
        assert result_dims is None

    def test_bounds_no_collapse(self):
        points = np.array([1, 2, 3])
        bounds = np.array([[0, 2], [1, 3], [2, 4]])
        result_dims, result_pts, result_bds = _reduce_points_and_bounds(
            points, (bounds[..., 0], bounds[..., 1])
        )
        assert_array_equal(result_pts, points)
        assert_array_equal(result_bds, bounds)
        assert result_dims == (0,)
