# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.PROPORTION` aggregator."""

import numpy.ma as ma
import pytest

from iris.analysis import PROPORTION
from iris.coords import DimCoord
import iris.cube
from iris.tests import _shared_utils


class Test_units_func:
    def test(self):
        assert PROPORTION.units_func is not None
        new_units = PROPORTION.units_func(None)
        assert new_units == 1


class Test_masked:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.cube.Cube(ma.masked_equal([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)
        self.func = lambda x: x >= 3

    def test_ma(self):
        cube = self.cube.collapsed("foo", PROPORTION, function=self.func)
        _shared_utils.assert_array_equal(cube.data, [0.5])

    def test_false_mask(self):
        # Test corner case where mask is returned as boolean value rather
        # than boolean array when the mask is unspecified on construction.
        masked_cube = iris.cube.Cube(ma.array([1, 2, 3, 4, 5]))
        masked_cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)
        cube = masked_cube.collapsed("foo", PROPORTION, function=self.func)
        _shared_utils.assert_array_equal(cube.data, ma.array([0.6]))


class Test_name:
    def test(self):
        assert PROPORTION.name() == "proportion"


class Test_aggregate_shape:
    def test(self):
        shape = ()
        kwargs = dict()
        assert PROPORTION.aggregate_shape(**kwargs) == shape
        kwargs = dict(captain="caveman", penelope="pitstop")
        assert PROPORTION.aggregate_shape(**kwargs) == shape
