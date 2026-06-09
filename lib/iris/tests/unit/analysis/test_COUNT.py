# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.COUNT` aggregator."""

import numpy as np
import numpy.ma as ma
import pytest

from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.analysis import COUNT
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import _shared_utils


class Test_basics:
    @pytest.fixture(autouse=True)
    def _setup(self):
        data = np.array([1, 2, 3, 4, 5])
        coord = DimCoord([6, 7, 8, 9, 10], long_name="foo")
        self.cube = Cube(data)
        self.cube.add_dim_coord(coord, 0)
        self.lazy_cube = Cube(as_lazy_data(data))
        self.lazy_cube.add_dim_coord(coord, 0)
        self.func = lambda x: x >= 3

    def test_name(self):
        assert COUNT.name() == "count"

    def test_no_function(self):
        exp_emsg = r"function must be a callable. Got <.* 'NoneType'>"
        with pytest.raises(TypeError, match=exp_emsg):
            COUNT.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)

    def test_not_callable(self):
        with pytest.raises(TypeError, match="function must be a callable"):
            COUNT.aggregate(self.cube.data, axis=0, function="wibble")

    def test_lazy_not_callable(self):
        with pytest.raises(TypeError, match="function must be a callable"):
            COUNT.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0, function="wibble")

    def test_collapse(self):
        data = COUNT.aggregate(self.cube.data, axis=0, function=self.func)
        _shared_utils.assert_array_equal(data, [3])

    def test_lazy(self):
        lazy_data = COUNT.lazy_aggregate(
            self.lazy_cube.lazy_data(), axis=0, function=self.func
        )
        assert is_lazy_data(lazy_data)

    def test_lazy_collapse(self):
        lazy_data = COUNT.lazy_aggregate(
            self.lazy_cube.lazy_data(), axis=0, function=self.func
        )
        _shared_utils.assert_array_equal(lazy_data.compute(), [3])


class Test_units_func:
    def test(self):
        assert COUNT.units_func is not None
        new_units = COUNT.units_func(None)
        assert new_units == 1


class Test_masked:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = Cube(ma.masked_equal([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)
        self.func = lambda x: x >= 3

    def test_ma(self):
        data = COUNT.aggregate(self.cube.data, axis=0, function=self.func)
        _shared_utils.assert_array_equal(data, [2])


class Test_lazy_masked:
    @pytest.fixture(autouse=True)
    def _setup(self):
        lazy_data = as_lazy_data(ma.masked_equal([1, 2, 3, 4, 5], 3))
        self.lazy_cube = Cube(lazy_data)
        self.lazy_cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)
        self.func = lambda x: x >= 3

    def test_ma(self):
        lazy_data = COUNT.lazy_aggregate(
            self.lazy_cube.lazy_data(), axis=0, function=self.func
        )
        assert is_lazy_data(lazy_data)
        _shared_utils.assert_array_equal(lazy_data.compute(), [2])


class Test_aggregate_shape:
    def test(self):
        shape = ()
        kwargs = dict()
        assert COUNT.aggregate_shape(**kwargs) == shape
        kwargs = dict(wibble="wobble")
        assert COUNT.aggregate_shape(**kwargs) == shape
