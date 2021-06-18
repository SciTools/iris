# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.STD_DEV` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
from iris.analysis import STD_DEV
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
        self.assertEqual(STD_DEV.name(), "standard_deviation")

    def test_collapse(self):
        data = STD_DEV.aggregate(self.cube.data, axis=0)
        self.assertArrayAlmostEqual(data, [1.58113883])

    def test_lazy(self):
        lazy_data = STD_DEV.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertTrue(is_lazy_data(lazy_data))

    def test_lazy_collapse(self):
        lazy_data = STD_DEV.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertArrayAlmostEqual(lazy_data.compute(), [1.58113883])


class Test_lazy_aggregate(tests.IrisTest):
    def test_mdtol(self):
        na = -999.888
        array = np.ma.masked_equal(
            [[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 3.0, na], [1.0, 2.0, na, na]], na
        )
        array = as_lazy_data(array)
        var = STD_DEV.lazy_aggregate(array, axis=1, mdtol=0.3)
        masked_result = as_concrete_data(var)
        masked_expected = np.ma.masked_array(
            [0.57735, 1.0, 0.707107], mask=[0, 0, 1]
        )
        self.assertMaskedArrayAlmostEqual(masked_result, masked_expected)

    def test_ddof_one(self):
        array = as_lazy_data(np.arange(8))
        var = STD_DEV.lazy_aggregate(array, axis=0, ddof=1)
        result = as_concrete_data(var)
        self.assertArrayAlmostEqual(result, np.array(2.449489))

    def test_ddof_zero(self):
        array = as_lazy_data(np.arange(8))
        var = STD_DEV.lazy_aggregate(array, axis=0, ddof=0)
        result = as_concrete_data(var)
        self.assertArrayAlmostEqual(result, np.array(2.291287))


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(STD_DEV.aggregate_shape(**kwargs), shape)
        kwargs = dict(forfar=5, fife=4)
        self.assertTupleEqual(STD_DEV.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
