# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.SUM` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.analysis import SUM
from iris.coords import DimCoord
from iris.cube import Cube


class Test_basics(tests.IrisTest):
    def setUp(self):
        data = np.array([1, 2, 3, 4, 5])
        coord = DimCoord([6, 7, 8, 9, 10], long_name="foo")
        self.cube = Cube(data)
        self.cube.add_dim_coord(coord, 0)
        self.lazy_cube = Cube(as_lazy_data(data))
        self.lazy_cube.add_dim_coord(coord, 0)

    def test_name(self):
        self.assertEqual(SUM.name(), "sum")

    def test_collapse(self):
        data = SUM.aggregate(self.cube.data, axis=0)
        self.assertArrayEqual(data, [15])

    def test_lazy(self):
        lazy_data = SUM.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertTrue(is_lazy_data(lazy_data))

    def test_lazy_collapse(self):
        lazy_data = SUM.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertArrayEqual(lazy_data.compute(), [15])


class Test_masked(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(ma.masked_equal([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)

    def test_ma(self):
        data = SUM.aggregate(self.cube.data, axis=0)
        self.assertArrayEqual(data, [12])


class Test_lazy_masked(tests.IrisTest):
    def setUp(self):
        masked_data = ma.masked_equal([1, 2, 3, 4, 5], 3)
        self.cube = Cube(as_lazy_data(masked_data))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)

    def test_lazy_ma(self):
        lazy_data = SUM.lazy_aggregate(self.cube.lazy_data(), axis=0)
        self.assertTrue(is_lazy_data(lazy_data))
        self.assertArrayEqual(lazy_data.compute(), [12])


class Test_weights_and_returned(tests.IrisTest):
    def setUp(self):
        data_2d = np.arange(1, 11).reshape(2, 5)
        coord_0 = DimCoord([11, 12], long_name="bar")
        coord_1 = DimCoord([6, 7, 8, 9, 10], long_name="foo")
        self.cube_2d = Cube(data_2d)
        self.cube_2d.add_dim_coord(coord_0, 0)
        self.cube_2d.add_dim_coord(coord_1, 1)
        self.weights = np.array([2, 1, 1, 1, 1] * 2).reshape(2, 5)

    def test_weights(self):
        data = SUM.aggregate(self.cube_2d.data, axis=0, weights=self.weights)
        self.assertArrayEqual(data, [14, 9, 11, 13, 15])

    def test_returned(self):
        data, weights = SUM.aggregate(self.cube_2d.data, axis=0, returned=True)
        self.assertArrayEqual(data, [7, 9, 11, 13, 15])
        self.assertArrayEqual(weights, [2, 2, 2, 2, 2])

    def test_weights_and_returned(self):
        data, weights = SUM.aggregate(
            self.cube_2d.data, axis=0, weights=self.weights, returned=True
        )
        self.assertArrayEqual(data, [14, 9, 11, 13, 15])
        self.assertArrayEqual(weights, [4, 2, 2, 2, 2])

    def test_masked_weights_and_returned(self):
        array = ma.array(
            self.cube_2d.data, mask=[[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
        )
        data, weights = SUM.aggregate(
            array, axis=0, weights=self.weights, returned=True
        )
        self.assertArrayEqual(data, [14, 9, 8, 4, 15])
        self.assertArrayEqual(weights, [4, 2, 1, 1, 2])


class Test_lazy_weights_and_returned(tests.IrisTest):
    def setUp(self):
        data_2d = np.arange(1, 11).reshape(2, 5)
        coord_0 = DimCoord([11, 12], long_name="bar")
        coord_1 = DimCoord([6, 7, 8, 9, 10], long_name="foo")
        self.cube_2d = Cube(as_lazy_data(data_2d))
        self.cube_2d.add_dim_coord(coord_0, 0)
        self.cube_2d.add_dim_coord(coord_1, 1)
        self.weights = np.array([2, 1, 1, 1, 1] * 2).reshape(2, 5)

    def test_weights(self):
        lazy_data = SUM.lazy_aggregate(
            self.cube_2d.lazy_data(), axis=0, weights=self.weights
        )
        self.assertTrue(is_lazy_data(lazy_data))
        self.assertArrayEqual(lazy_data.compute(), [14, 9, 11, 13, 15])

    def test_returned(self):
        lazy_data, weights = SUM.lazy_aggregate(
            self.cube_2d.lazy_data(), axis=0, returned=True
        )
        self.assertTrue(is_lazy_data(lazy_data))
        self.assertArrayEqual(lazy_data.compute(), [7, 9, 11, 13, 15])
        self.assertArrayEqual(weights, [2, 2, 2, 2, 2])

    def test_weights_and_returned(self):
        lazy_data, weights = SUM.lazy_aggregate(
            self.cube_2d.lazy_data(),
            axis=0,
            weights=self.weights,
            returned=True,
        )
        self.assertTrue(is_lazy_data(lazy_data))
        self.assertArrayEqual(lazy_data.compute(), [14, 9, 11, 13, 15])
        self.assertArrayEqual(weights, [4, 2, 2, 2, 2])

    def test_masked_weights_and_returned(self):
        array = da.ma.masked_array(
            self.cube_2d.lazy_data(), mask=[[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
        )
        lazy_data, weights = SUM.lazy_aggregate(
            array, axis=0, weights=self.weights, returned=True
        )
        self.assertTrue(is_lazy_data(lazy_data))
        self.assertArrayEqual(lazy_data.compute(), [14, 9, 8, 4, 15])
        self.assertArrayEqual(weights, [4, 2, 1, 1, 2])


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(SUM.aggregate_shape(**kwargs), shape)
        kwargs = dict(wibble="wobble")
        self.assertTupleEqual(SUM.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
