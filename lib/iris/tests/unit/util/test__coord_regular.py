# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test elements of :mod:`iris.util` that deal with checking coord regularity.
Specifically, this module tests the following functions:

  * :func:`iris.util.is_regular`,
  * :func:`iris.util.regular_step`, and
  * :func:`iris.util.points_step`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.exceptions import CoordinateMultiDimError, CoordinateNotRegularError
from iris.util import is_regular, points_step, regular_step


class Test_is_regular(tests.IrisTest):
    def test_coord_with_regular_step(self):
        coord = DimCoord(np.arange(5))
        result = is_regular(coord)
        self.assertTrue(result)

    def test_coord_with_irregular_step(self):
        # Check that a `CoordinateNotRegularError` is captured.
        coord = AuxCoord(np.array([2, 5, 1, 4]))
        result = is_regular(coord)
        self.assertFalse(result)

    def test_scalar_coord(self):
        # Check that a `ValueError` is captured.
        coord = DimCoord(5)
        result = is_regular(coord)
        self.assertFalse(result)

    def test_coord_with_string_points(self):
        # Check that a `TypeError` is captured.
        coord = AuxCoord(["a", "b", "c"])
        result = is_regular(coord)
        self.assertFalse(result)


class Test_regular_step(tests.IrisTest):
    def test_basic(self):
        dtype = np.float64
        points = np.arange(5, dtype=dtype)
        coord = DimCoord(points)
        expected = np.mean(np.diff(points))
        result = regular_step(coord)
        self.assertEqual(expected, result)
        self.assertEqual(result.dtype, dtype)

    def test_2d_coord(self):
        coord = AuxCoord(np.arange(8).reshape(2, 4))
        exp_emsg = "Expected 1D coord"
        with self.assertRaisesRegex(CoordinateMultiDimError, exp_emsg):
            regular_step(coord)

    def test_scalar_coord(self):
        coord = DimCoord(5)
        exp_emsg = "non-scalar coord"
        with self.assertRaisesRegex(ValueError, exp_emsg):
            regular_step(coord)

    def test_coord_with_irregular_step(self):
        name = "latitude"
        coord = AuxCoord(np.array([2, 5, 1, 4]), standard_name=name)
        exp_emsg = "{} is not regular".format(name)
        with self.assertRaisesRegex(CoordinateNotRegularError, exp_emsg):
            regular_step(coord)


class Test_points_step(tests.IrisTest):
    def test_regular_points(self):
        regular_points = np.arange(5)
        exp_avdiff = np.mean(np.diff(regular_points))
        result_avdiff, result = points_step(regular_points)
        self.assertEqual(exp_avdiff, result_avdiff)
        self.assertTrue(result)

    def test_irregular_points(self):
        irregular_points = np.array([2, 5, 1, 4])
        exp_avdiff = np.mean(np.diff(irregular_points))
        result_avdiff, result = points_step(irregular_points)
        self.assertEqual(exp_avdiff, result_avdiff)
        self.assertFalse(result)

    def test_single_point(self):
        lone_point = np.array([4])
        result_avdiff, result = points_step(lone_point)
        self.assertTrue(np.isnan(result_avdiff))
        self.assertTrue(result)

    def test_no_points(self):
        no_points = np.array([])
        result_avdiff, result = points_step(no_points)
        self.assertTrue(np.isnan(result_avdiff))
        self.assertTrue(result)


if __name__ == "__main__":
    tests.main()
