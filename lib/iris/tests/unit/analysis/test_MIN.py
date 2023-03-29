# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.MIN` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_lazy_data, is_lazy_data
from iris.analysis import MIN
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
        self.assertEqual(MIN.name(), "minimum")

    def test_collapse(self):
        data = MIN.aggregate(self.cube.data, axis=0)
        self.assertArrayEqual(data, [1])

    def test_lazy(self):
        lazy_data = MIN.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertTrue(is_lazy_data(lazy_data))

    def test_lazy_collapse(self):
        lazy_data = MIN.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertArrayEqual(lazy_data.compute(), [1])


class Test_masked(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(ma.masked_less([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)

    def test_ma(self):
        data = MIN.aggregate(self.cube.data, axis=0)
        self.assertArrayEqual(data, [3])


class Test_lazy_masked(tests.IrisTest):
    def setUp(self):
        masked_data = ma.masked_less([1, 2, 3, 4, 5], 3)
        self.cube = Cube(as_lazy_data(masked_data))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name="foo"), 0)

    def test_lazy_ma(self):
        lazy_data = MIN.lazy_aggregate(self.cube.lazy_data(), axis=0)
        self.assertTrue(is_lazy_data(lazy_data))
        self.assertArrayEqual(lazy_data.compute(), [3])


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(MIN.aggregate_shape(**kwargs), shape)
        kwargs = dict(wibble="wobble")
        self.assertTupleEqual(MIN.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
