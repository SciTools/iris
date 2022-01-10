# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.VARIANCE` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data
from iris.analysis import VARIANCE
from iris.coords import DimCoord
import iris.cube


class Test_units_func(tests.IrisTest):
    def test(self):
        self.assertIsNotNone(VARIANCE.units_func)
        mul = mock.Mock(return_value=mock.sentinel.new_unit)
        units = mock.Mock(__mul__=mul)
        new_units = VARIANCE.units_func(units)
        # Make sure the VARIANCE units_func tries to square the units.
        mul.assert_called_once_with(units)
        self.assertEqual(new_units, mock.sentinel.new_unit)


class Test_masked(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube(ma.masked_equal([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)

    def test_ma_ddof0(self):
        cube = self.cube.collapsed("foo", VARIANCE, ddof=0)
        expected = 10 / 4.0
        self.assertArrayEqual(np.var(self.cube.data, ddof=0), expected)
        self.assertArrayAlmostEqual(cube.data, expected)

    def test_ma_ddof1(self):
        cube = self.cube.collapsed("foo", VARIANCE, ddof=1)
        expected = 10 / 3.0
        self.assertArrayEqual(np.var(self.cube.data, ddof=1), expected)
        self.assertArrayEqual(cube.data, expected)

        # test that the default ddof is 1
        default_cube = self.cube.collapsed("foo", VARIANCE)
        self.assertArrayEqual(cube.data, default_cube.data)


class Test_lazy_aggregate(tests.IrisTest):
    def test_ddof_one(self):
        array = as_lazy_data(np.arange(8))
        var = VARIANCE.lazy_aggregate(array, axis=0, ddof=1)
        result = as_concrete_data(var)
        self.assertArrayAlmostEqual(result, np.array(6.0))

    def test_ddof_zero(self):
        array = as_lazy_data(np.arange(8))
        var = VARIANCE.lazy_aggregate(array, axis=0, ddof=0)
        result = as_concrete_data(var)
        self.assertArrayAlmostEqual(result, np.array(5.25))


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(VARIANCE.name(), "variance")


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(VARIANCE.aggregate_shape(**kwargs), shape)
        kwargs = dict(bat="man", wonder="woman")
        self.assertTupleEqual(VARIANCE.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
