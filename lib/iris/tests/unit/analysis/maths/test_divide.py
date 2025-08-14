# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.analysis.maths.divide` function."""

import operator

import numpy as np
import pytest

from iris.analysis.maths import divide
from iris.cube import Cube
from iris.tests import _shared_utils
from iris.tests.unit.analysis.maths import (
    CubeArithmeticBroadcastingTestMixin,
    CubeArithmeticCoordsTest,
    CubeArithmeticMaskingTestMixin,
)


@_shared_utils.skip_data
class TestBroadcasting(CubeArithmeticBroadcastingTestMixin):
    @property
    def data_op(self):
        return operator.truediv

    @property
    def cube_func(self):
        return divide


class TestMasking(CubeArithmeticMaskingTestMixin):
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

        _shared_utils.assert_array_equal(com, res)

    def test_masked_div_zero(self):
        # Ensure cube behaviour matches numpy operator behaviour for the
        # handling of arrays containing 0.
        dat_a = np.ma.array([0.0, 0.0, 0.0, 0.0], mask=False)
        dat_b = np.ma.array([2.0, 2.0, 2.0, 2.0], mask=False)

        cube_a = Cube(dat_a)
        cube_b = Cube(dat_b)

        com = self.data_op(dat_b, dat_a)
        res = self.cube_func(cube_b, cube_a).data

        _shared_utils.assert_masked_array_equal(com, res, strict=True)


class TestCoordMatch(CubeArithmeticCoordsTest):
    def test_no_match(self):
        cube1, cube2 = self.setup_non_matching()
        expected = "Insufficient matching coordinate metadata to resolve cubes"
        with pytest.raises(ValueError, match=expected):
            divide(cube1, cube2)

    def test_reversed_points(self):
        cube1, cube2 = self.setup_reversed()
        expected = "Coordinate '.*' has different points"
        with pytest.raises(ValueError, match=expected):
            divide(cube1, cube2)
