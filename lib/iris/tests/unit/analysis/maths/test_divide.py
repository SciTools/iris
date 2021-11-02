# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :func:`iris.analysis.maths.divide` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import operator

import numpy as np

from iris.analysis.maths import divide
from iris.cube import Cube
from iris.tests.unit.analysis.maths import (
    CubeArithmeticBroadcastingTestMixin,
    CubeArithmeticCoordsTest,
    CubeArithmeticMaskingTestMixin,
)


@tests.skip_data
@tests.iristest_timing_decorator
class TestBroadcasting(
    tests.IrisTest_nometa, CubeArithmeticBroadcastingTestMixin
):
    @property
    def data_op(self):
        return operator.truediv

    @property
    def cube_func(self):
        return divide


@tests.iristest_timing_decorator
class TestMasking(tests.IrisTest_nometa, CubeArithmeticMaskingTestMixin):
    @property
    def data_op(self):
        return operator.truediv

    @property
    def cube_func(self):
        return divide

    def test_unmasked_div_zero(self):
        # Ensure cube behaviour matches numpy operator behaviour for the
        # handling of arrays containing 0.
        dat_a = np.array([0.0, 0.0, 0.0, 0.0])
        dat_b = np.array([2.0, 2.0, 2.0, 2.0])

        cube_a = Cube(dat_a)
        cube_b = Cube(dat_b)

        com = self.data_op(dat_b, dat_a)
        res = self.cube_func(cube_b, cube_a).data

        self.assertArrayEqual(com, res)

    def test_masked_div_zero(self):
        # Ensure cube behaviour matches numpy operator behaviour for the
        # handling of arrays containing 0.
        dat_a = np.ma.array([0.0, 0.0, 0.0, 0.0], mask=False)
        dat_b = np.ma.array([2.0, 2.0, 2.0, 2.0], mask=False)

        cube_a = Cube(dat_a)
        cube_b = Cube(dat_b)

        com = self.data_op(dat_b, dat_a)
        res = self.cube_func(cube_b, cube_a).data

        self.assertMaskedArrayEqual(com, res, strict=True)


class TestCoordMatch(CubeArithmeticCoordsTest):
    def test_no_match(self):
        cube1, cube2 = self.SetUpNonMatching()
        with self.assertRaises(ValueError):
            divide(cube1, cube2)

    def test_reversed_points(self):
        cube1, cube2 = self.SetUpReversed()
        with self.assertRaises(ValueError):
            divide(cube1, cube2)


if __name__ == "__main__":
    tests.main()
