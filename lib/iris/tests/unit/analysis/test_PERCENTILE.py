# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.PERCENTILE` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris.analysis import PERCENTILE


class Test_aggregate(tests.IrisTest):
    def test_missing_mandatory_kwarg(self):
        emsg = "percentile aggregator requires .* keyword argument 'percent'"
        with self.assertRaisesRegex(ValueError, emsg):
            PERCENTILE.aggregate("dummy", axis=0)

    def test_1d_single(self):
        data = np.arange(11)
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        expected = 5
        self.assertTupleEqual(actual.shape, ())
        self.assertEqual(actual, expected)

    def test_masked_1d_single(self):
        data = ma.arange(11)
        data[3:7] = ma.masked
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        expected = 7
        self.assertTupleEqual(actual.shape, ())
        self.assertEqual(actual, expected)

    def test_1d_multi(self):
        data = np.arange(11)
        percent = np.array([20, 50, 90])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        expected = [2, 5, 9]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertArrayEqual(actual, expected)

    def test_masked_1d_multi(self):
        data = ma.arange(11)
        data[3:9] = ma.masked
        percent = np.array([25, 50, 75])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        expected = [1, 2, 9]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertArrayEqual(actual, expected)

    def test_2d_single(self):
        shape = (2, 11)
        data = np.arange(np.prod(shape)).reshape(shape)
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        self.assertTupleEqual(actual.shape, shape[-1:])
        expected = np.arange(shape[-1]) + 5.5
        self.assertArrayEqual(actual, expected)

    def test_masked_2d_single(self):
        shape = (2, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data[1, 1::2] = ma.masked
        actual = PERCENTILE.aggregate(data, axis=0, percent=50)
        self.assertTupleEqual(actual.shape, shape[-1:])
        expected = np.empty(shape[-1:])
        expected[1::2] = data[0, 1::2]
        expected[::2] = data[1, ::2]
        self.assertArrayEqual(actual, expected)

    def test_2d_multi(self):
        shape = (2, 10)
        data = np.arange(np.prod(shape)).reshape(shape)
        percent = np.array([10, 50, 90, 100])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        expected = np.tile(np.arange(shape[-1]), percent.size)
        expected = expected.reshape(percent.size, shape[-1]).T + 1
        expected = expected + (percent / 10 - 1)
        self.assertArrayAlmostEqual(actual, expected)

    def test_masked_2d_multi(self):
        shape = (3, 10)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[1] = ma.masked
        percent = np.array([10, 50, 70, 80])
        actual = PERCENTILE.aggregate(data, axis=0, percent=percent)
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        expected = np.tile(np.arange(shape[-1]), percent.size)
        expected = expected.reshape(percent.size, shape[-1]).T
        expected = expected + (percent / 10 * 2)
        self.assertArrayAlmostEqual(actual, expected)


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(PERCENTILE.name(), "percentile")


class Test_aggregate_shape(tests.IrisTest):
    def test_missing_mandatory_kwarg(self):
        emsg = "percentile aggregator requires .* keyword argument 'percent'"
        with self.assertRaisesRegex(ValueError, emsg):
            PERCENTILE.aggregate_shape()
        with self.assertRaisesRegex(ValueError, emsg):
            kwargs = dict()
            PERCENTILE.aggregate_shape(**kwargs)
        with self.assertRaisesRegex(ValueError, emsg):
            kwargs = dict(point=10)
            PERCENTILE.aggregate_shape(**kwargs)

    def test_mandatory_kwarg_no_shape(self):
        kwargs = dict(percent=50)
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), ())
        kwargs = dict(percent=[50])
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), ())

    def test_mandatory_kwarg_shape(self):
        kwargs = dict(percent=(10, 20))
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), (2,))
        kwargs = dict(percent=list(range(13)))
        self.assertTupleEqual(PERCENTILE.aggregate_shape(**kwargs), (13,))


class Test_cell_method(tests.IrisTest):
    def test(self):
        self.assertIsNone(PERCENTILE.cell_method)


if __name__ == "__main__":
    tests.main()
