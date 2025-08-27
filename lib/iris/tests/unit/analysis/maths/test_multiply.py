# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.analysis.maths.multiply` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import operator

import pytest

from iris.analysis.maths import multiply
from iris.tests import _shared_utils
from iris.tests.unit.analysis.maths import (
    CubeArithmeticBroadcastingTestMixin,
    CubeArithmeticCoordsTest,
    CubeArithmeticMaskedConstantTestMixin,
    CubeArithmeticMaskingTestMixin,
)


@_shared_utils.skip_data
class TestBroadcasting(CubeArithmeticBroadcastingTestMixin):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


class TestMasking(CubeArithmeticMaskingTestMixin):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


class TestCoordMatch(CubeArithmeticCoordsTest):
    def test_no_match(self):
        cube1, cube2 = self.setup_non_matching()
        expected = "Insufficient matching coordinate metadata to resolve cubes"
        with pytest.raises(ValueError, match=expected):
            multiply(cube1, cube2)

    def test_reversed_points(self):
        cube1, cube2 = self.setup_reversed()
        expected = "Coordinate '.*' has different points"
        with pytest.raises(ValueError, match=expected):
            multiply(cube1, cube2)


class TestMaskedConstant(CubeArithmeticMaskedConstantTestMixin):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply
