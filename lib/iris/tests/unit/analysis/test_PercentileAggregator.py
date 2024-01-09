# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.analysis.PercentileAggregator` class instance."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import dask.array as da
import numpy as np

from iris._lazy_data import as_concrete_data
from iris.analysis import PercentileAggregator
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


class Test(tests.IrisTest):
    def test_init(self):
        name = "percentile"
        units_func = mock.sentinel.units_func
        aggregator = PercentileAggregator(units_func=units_func)
        self.assertEqual(aggregator.name(), name)
        self.assertIs(aggregator.units_func, units_func)
        self.assertIsNone(aggregator.cell_method)


class Test_post_process(tests.IrisTest):
    def setUp(self):
        shape = (2, 5)
        data = np.arange(np.prod(shape))

        self.coord_simple = DimCoord(data, "time")
        self.cube_simple = Cube(data)
        self.cube_simple.add_dim_coord(self.coord_simple, 0)

        self.coord_multi_0 = DimCoord(np.arange(shape[0]), "time")
        self.coord_multi_1 = DimCoord(np.arange(shape[1]), "height")
        self.cube_multi = Cube(data.reshape(shape))
        self.cube_multi.add_dim_coord(self.coord_multi_0, 0)
        self.cube_multi.add_dim_coord(self.coord_multi_1, 1)

    def test_missing_mandatory_kwarg(self):
        aggregator = PercentileAggregator()
        emsg = "percentile aggregator requires .* keyword argument 'percent'"
        with self.assertRaisesRegex(ValueError, emsg):
            aggregator.aggregate("dummy", axis=0)

    def test_simple_single_point(self):
        aggregator = PercentileAggregator()
        percent = 50
        kwargs = dict(percent=percent)
        data = np.empty(self.cube_simple.shape)
        coords = [self.coord_simple]
        actual = aggregator.post_process(self.cube_simple, data, coords, **kwargs)
        self.assertEqual(actual.shape, self.cube_simple.shape)
        self.assertIs(actual.data, data)
        name = "percentile_over_time"
        coord = actual.coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)

    def test_simple_multiple_points(self):
        aggregator = PercentileAggregator()
        percent = np.array([10, 20, 50, 90])
        kwargs = dict(percent=percent)
        shape = self.cube_simple.shape + percent.shape
        data = np.empty(shape)
        coords = [self.coord_simple]
        actual = aggregator.post_process(self.cube_simple, data, coords, **kwargs)
        self.assertEqual(actual.shape, percent.shape + self.cube_simple.shape)
        expected = data.T
        self.assertArrayEqual(actual.data, expected)
        name = "percentile_over_time"
        coord = actual.coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)

    def test_multi_single_point(self):
        aggregator = PercentileAggregator()
        percent = 70
        kwargs = dict(percent=percent)
        data = np.empty(self.cube_multi.shape)
        coords = [self.coord_multi_0]
        actual = aggregator.post_process(self.cube_multi, data, coords, **kwargs)
        self.assertEqual(actual.shape, self.cube_multi.shape)
        self.assertIs(actual.data, data)
        name = "percentile_over_time"
        coord = actual.coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)

    def test_multi_multiple_points(self):
        aggregator = PercentileAggregator()
        percent = np.array([17, 29, 81])
        kwargs = dict(percent=percent)
        shape = self.cube_multi.shape + percent.shape
        data = np.empty(shape)
        coords = [self.coord_multi_0]
        actual = aggregator.post_process(self.cube_multi, data, coords, **kwargs)
        self.assertEqual(actual.shape, percent.shape + self.cube_multi.shape)
        expected = np.moveaxis(data, -1, 0)
        self.assertArrayEqual(actual.data, expected)
        name = "percentile_over_time"
        coord = actual.coord(name)
        expected = AuxCoord(percent, long_name=name, units="percent")
        self.assertEqual(coord, expected)

    def test_multi_multiple_points_lazy(self):
        # Check that lazy data is preserved.
        aggregator = PercentileAggregator()
        percent = np.array([17, 29, 81])
        kwargs = dict(percent=percent)
        shape = self.cube_multi.shape + percent.shape
        data = da.arange(np.prod(shape)).reshape(shape)
        coords = [self.coord_multi_0]
        actual = aggregator.post_process(self.cube_multi, data, coords, **kwargs)
        self.assertEqual(actual.shape, percent.shape + self.cube_multi.shape)
        self.assertTrue(actual.has_lazy_data())
        expected = np.moveaxis(as_concrete_data(data), -1, 0)
        self.assertArrayEqual(actual.data, expected)


if __name__ == "__main__":
    tests.main()
