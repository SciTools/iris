# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.analysis.PercentileAggregator` class instance.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

import numpy as np

from iris.analysis import WeightedPercentileAggregator, _weighted_percentile
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


class Test(tests.IrisTest):
    def test_init(self):
        name = "weighted_percentile"
        call_func = _weighted_percentile
        units_func = mock.sentinel.units_func
        lazy_func = mock.sentinel.lazy_func
        aggregator = WeightedPercentileAggregator(
            units_func=units_func, lazy_func=lazy_func
        )
        self.assertEqual(aggregator.name(), name)
        self.assertIs(aggregator.call_func, call_func)
        self.assertIs(aggregator.units_func, units_func)
        self.assertIs(aggregator.lazy_func, lazy_func)
        self.assertIsNone(aggregator.cell_method)


class Test_post_process(tests.IrisTest):
    def setUp(self):
        shape = (2, 5)
        data = np.arange(np.prod(shape))

        self.coord_simple = DimCoord(data, "time")
        self.cube_simple = Cube(data)
        self.cube_simple.add_dim_coord(self.coord_simple, 0)
        self.weights_simple = np.ones_like(data, dtype=float)

        self.coord_multi_0 = DimCoord(np.arange(shape[0]), "time")
        self.coord_multi_1 = DimCoord(np.arange(shape[1]), "height")
        self.cube_multi = Cube(data.reshape(shape))
        self.cube_multi.add_dim_coord(self.coord_multi_0, 0)
        self.cube_multi.add_dim_coord(self.coord_multi_1, 1)
        self.weights_multi = np.ones(shape, dtype=float)

    def test_missing_mandatory_kwarg(self):
        aggregator = WeightedPercentileAggregator()
        emsg = (
            "weighted_percentile aggregator requires "
            ".* keyword argument 'percent'"
        )
        with self.assertRaisesRegex(ValueError, emsg):
            aggregator.aggregate("dummy", axis=0, weights=None)
        emsg = (
            "weighted_percentile aggregator requires "
            ".* keyword argument 'weights'"
        )
        with self.assertRaisesRegex(ValueError, emsg):
            aggregator.aggregate("dummy", axis=0, percent=50)

    def test_simple_single_point(self):
        aggregator = WeightedPercentileAggregator()
        percent = 50
        kwargs = dict(percent=percent, weights=self.weights_simple)
        data = np.empty(self.cube_simple.shape)
        coords = [self.coord_simple]
        actual = aggregator.post_process(
            self.cube_simple, data, coords, **kwargs
        )
        self.assertEqual(actual.shape, self.cube_simple.shape)
        self.assertIs(actual.data, data)
        name = "weighted_percentile_over_time"
        coord = actual.coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)

    def test_simple_multiple_points(self):
        aggregator = WeightedPercentileAggregator()
        percent = np.array([10, 20, 50, 90])
        kwargs = dict(
            percent=percent, weights=self.weights_simple, returned=True
        )
        shape = self.cube_simple.shape + percent.shape
        data = np.empty(shape)
        total_weights = 1.0
        coords = [self.coord_simple]
        actual = aggregator.post_process(
            self.cube_simple, (data, total_weights), coords, **kwargs
        )
        self.assertEqual(len(actual), 2)
        self.assertEqual(
            actual[0].shape, percent.shape + self.cube_simple.shape
        )
        expected = np.rollaxis(data, -1)
        self.assertArrayEqual(actual[0].data, expected)
        self.assertIs(actual[1], total_weights)
        name = "weighted_percentile_over_time"
        coord = actual[0].coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)

    def test_multi_single_point(self):
        aggregator = WeightedPercentileAggregator()
        percent = 70
        kwargs = dict(percent=percent, weights=self.weights_multi)
        data = np.empty(self.cube_multi.shape)
        coords = [self.coord_multi_0]
        actual = aggregator.post_process(
            self.cube_multi, data, coords, **kwargs
        )
        self.assertEqual(actual.shape, self.cube_multi.shape)
        self.assertIs(actual.data, data)
        name = "weighted_percentile_over_time"
        coord = actual.coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)

    def test_multi_multiple_points(self):
        aggregator = WeightedPercentileAggregator()
        percent = np.array([17, 29, 81])
        kwargs = dict(percent=percent, weights=self.weights_multi)
        shape = self.cube_multi.shape + percent.shape
        data = np.empty(shape)
        coords = [self.coord_multi_0]
        actual = aggregator.post_process(
            self.cube_multi, data, coords, **kwargs
        )
        self.assertEqual(actual.shape, percent.shape + self.cube_multi.shape)
        expected = np.rollaxis(data, -1)
        self.assertArrayEqual(actual.data, expected)
        name = "weighted_percentile_over_time"
        coord = actual.coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)


if __name__ == "__main__":
    tests.main()
