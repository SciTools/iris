# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :func:`iris.analysis.maths.subtract` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import operator

from iris.analysis.maths import subtract
from iris.tests.unit.analysis.maths import (
    CubeArithmeticBroadcastingTestMixin,
    CubeArithmeticCoordsTest,
    CubeArithmeticMaskedConstantTestMixin,
    CubeArithmeticMaskingTestMixin,
)


@tests.skip_data
class TestBroadcasting(tests.IrisTest, CubeArithmeticBroadcastingTestMixin):
    @property
    def data_op(self):
        return operator.sub

    @property
    def cube_func(self):
        return subtract


class TestMasking(tests.IrisTest, CubeArithmeticMaskingTestMixin):
    @property
    def data_op(self):
        return operator.sub

    @property
    def cube_func(self):
        return subtract


class TestCoordMatch(CubeArithmeticCoordsTest):
    def test_no_match(self):
        cube1, cube2 = self.SetUpNonMatching()
        with self.assertRaises(ValueError):
            subtract(cube1, cube2)

    def test_reversed_points(self):
        cube1, cube2 = self.SetUpReversed()
        with self.assertRaises(ValueError):
            subtract(cube1, cube2)


class TestMaskedConstant(
    tests.IrisTest, CubeArithmeticMaskedConstantTestMixin
):
    @property
    def data_op(self):
        return operator.sub

    @property
    def cube_func(self):
        return subtract


if __name__ == "__main__":
    tests.main()
