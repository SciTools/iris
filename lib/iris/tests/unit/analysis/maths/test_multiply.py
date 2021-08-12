# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :func:`iris.analysis.maths.multiply` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import operator

from iris.analysis.maths import multiply
from iris.tests.unit.analysis.maths import (
    CubeArithmeticBroadcastingTestMixin,
    CubeArithmeticCoordsTest,
    CubeArithmeticMaskedConstantTestMixin,
    CubeArithmeticMaskingTestMixin,
)


@tests.skip_data
@tests.iristest_timing_decorator
class TestBroadcasting(
    tests.IrisTest_nometa, CubeArithmeticBroadcastingTestMixin
):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


@tests.iristest_timing_decorator
class TestMasking(tests.IrisTest_nometa, CubeArithmeticMaskingTestMixin):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


class TestCoordMatch(CubeArithmeticCoordsTest):
    def test_no_match(self):
        cube1, cube2 = self.SetUpNonMatching()
        with self.assertRaises(ValueError):
            multiply(cube1, cube2)

    def test_reversed_points(self):
        cube1, cube2 = self.SetUpReversed()
        with self.assertRaises(ValueError):
            multiply(cube1, cube2)


@tests.iristest_timing_decorator
class TestMaskedConstant(
    tests.IrisTest_nometa, CubeArithmeticMaskedConstantTestMixin
):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


if __name__ == "__main__":
    tests.main()
