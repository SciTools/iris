# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.VARIANCE` aggregator."""

import numpy as np
import numpy.ma as ma
import pytest

from iris._lazy_data import as_concrete_data, as_lazy_data
from iris.analysis import VARIANCE
from iris.coords import DimCoord
import iris.cube
from iris.tests import _shared_utils


class Test_units_func:
    def test(self, mocker):
        assert VARIANCE.units_func is not None
        mul = mocker.Mock(return_value=mocker.sentinel.new_unit)
        units = mocker.Mock(__mul__=mul)
        new_units = VARIANCE.units_func(units)
        # Make sure the VARIANCE units_func tries to square the units.
        mul.assert_called_once_with(units)
        assert new_units == mocker.sentinel.new_unit


class Test_masked:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.cube.Cube(ma.masked_equal([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)

    def test_ma_ddof0(self):
        cube = self.cube.collapsed("foo", VARIANCE, ddof=0)
        expected = 10 / 4.0
        _shared_utils.assert_array_equal(np.var(self.cube.data, ddof=0), expected)
        _shared_utils.assert_array_almost_equal(cube.data, expected)

    def test_ma_ddof1(self):
        cube = self.cube.collapsed("foo", VARIANCE, ddof=1)
        expected = 10 / 3.0
        _shared_utils.assert_array_equal(np.var(self.cube.data, ddof=1), expected)
        _shared_utils.assert_array_equal(cube.data, expected)

        # test that the default ddof is 1
        default_cube = self.cube.collapsed("foo", VARIANCE)
        _shared_utils.assert_array_equal(cube.data, default_cube.data)


class Test_lazy_aggregate:
    def test_ddof_one(self):
        array = as_lazy_data(np.arange(8))
        var = VARIANCE.lazy_aggregate(array, axis=0, ddof=1)
        result = as_concrete_data(var)
        _shared_utils.assert_array_almost_equal(result, np.array(6.0))

    def test_ddof_zero(self):
        array = as_lazy_data(np.arange(8))
        var = VARIANCE.lazy_aggregate(array, axis=0, ddof=0)
        result = as_concrete_data(var)
        _shared_utils.assert_array_almost_equal(result, np.array(5.25))


class Test_name:
    def test(self):
        assert VARIANCE.name() == "variance"


class Test_aggregate_shape:
    def test(self):
        shape = ()
        kwargs = dict()
        assert VARIANCE.aggregate_shape(**kwargs) == shape
        kwargs = dict(bat="man", wonder="woman")
        assert VARIANCE.aggregate_shape(**kwargs) == shape
