# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.PERCENTILE` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import re

import numpy as np
import numpy.ma as ma

from iris.analysis import WPERCENTILE


class Test_aggregate(tests.IrisTest):
    def test_missing_mandatory_kwargs(self):
        emsg = "weighted_percentile aggregator requires .* keyword argument 'percent'"
        with self.assertRaisesRegex(ValueError, emsg):
            WPERCENTILE.aggregate("dummy", axis=0, weights=None)
        emsg = "weighted_percentile aggregator requires .* keyword argument 'weights'"
        with self.assertRaisesRegex(ValueError, emsg):
            WPERCENTILE.aggregate("dummy", axis=0, percent=50)

    def test_wrong_weights_shape(self):
        data = np.arange(11)
        weights = np.ones(10)
        emsg = re.escape(
            "For data array of shape (11,), weights should be (11,) or (11,)"
        )
        with self.assertRaisesRegex(ValueError, emsg):
            WPERCENTILE.aggregate(data, axis=0, percent=50, weights=weights)

    def test_1d_single(self):
        data = np.arange(11)
        weights = np.ones(data.shape)
        actual = WPERCENTILE.aggregate(data, axis=0, percent=50, weights=weights)
        expected = 5
        self.assertTupleEqual(actual.shape, ())
        self.assertEqual(actual, expected)

    def test_1d_single_unequal(self):
        data = np.arange(12)
        weights = np.ones(data.shape)
        weights[0:3] = 3
        actual, weight_total = WPERCENTILE.aggregate(
            data, axis=0, percent=50, weights=weights, returned=True
        )
        expected = 2.75
        self.assertTupleEqual(actual.shape, ())
        self.assertEqual(actual, expected)
        self.assertEqual(weight_total, 18)

    def test_masked_1d_single(self):
        data = ma.arange(11)
        weights = np.ones(data.shape)
        data[3:7] = ma.masked
        actual = WPERCENTILE.aggregate(data, axis=0, percent=50, weights=weights)
        expected = 7
        self.assertTupleEqual(actual.shape, ())
        self.assertEqual(actual, expected)

    def test_1d_multi(self):
        data = np.arange(11)
        weights = np.ones(data.shape)
        percent = np.array([20, 50, 90])
        actual = WPERCENTILE.aggregate(data, axis=0, percent=percent, weights=weights)
        expected = [1.7, 5, 9.4]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertArrayAlmostEqual(actual, expected)

    def test_1d_multi_unequal(self):
        data = np.arange(13)
        weights = np.ones(data.shape)
        weights[1::2] = 3
        percent = np.array([20, 50, 96])
        actual = WPERCENTILE.aggregate(data, axis=0, percent=percent, weights=weights)
        expected = [2.25, 6, 11.75]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertArrayAlmostEqual(actual, expected)

    def test_masked_1d_multi(self):
        data = ma.arange(11)
        weights = np.ones(data.shape)
        data[3:9] = ma.masked
        percent = np.array([25, 50, 75])
        actual = WPERCENTILE.aggregate(data, axis=0, percent=percent, weights=weights)
        expected = [0.75, 2, 9.25]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertArrayAlmostEqual(actual, expected)

    def test_2d_single(self):
        shape = (2, 11)
        data = np.arange(np.prod(shape)).reshape(shape)
        weights = np.ones(shape)
        actual = WPERCENTILE.aggregate(data, axis=0, percent=50, weights=weights)
        self.assertTupleEqual(actual.shape, shape[-1:])
        expected = np.arange(shape[-1]) + 5.5
        self.assertArrayEqual(actual, expected)

    def test_masked_2d_single(self):
        shape = (2, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data[1, 1::2] = ma.masked
        weights = np.ones(shape)
        actual = WPERCENTILE.aggregate(data, axis=0, percent=50, weights=weights)
        self.assertTupleEqual(actual.shape, shape[-1:])
        expected = np.empty(shape[-1:])
        expected[1::2] = data[0, 1::2]
        expected[::2] = data[1, ::2]
        self.assertArrayEqual(actual, expected)

    def test_2d_multi(self):
        shape = (2, 10)
        data = np.arange(np.prod(shape)).reshape(shape)
        weights = np.ones(shape)
        percent = np.array([10, 50, 70, 100])
        actual = WPERCENTILE.aggregate(data, axis=0, percent=percent, weights=weights)
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        expected = np.tile(np.arange(shape[-1]), percent.size).astype("f8")
        expected = expected.reshape(percent.size, shape[-1]).T
        expected[:, 1:-1] += (percent[1:-1] - 25) * 0.2
        expected[:, -1] += 10.0
        self.assertArrayAlmostEqual(actual, expected)

    def test_masked_2d_multi(self):
        shape = (3, 10)
        data = ma.arange(np.prod(shape)).reshape(shape)
        weights = np.ones(shape)
        data[1] = ma.masked
        percent = np.array([10, 50, 70, 80])
        actual = WPERCENTILE.aggregate(data, axis=0, percent=percent, weights=weights)
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        expected = np.tile(np.arange(shape[-1]), percent.size).astype("f8")
        expected = expected.reshape(percent.size, shape[-1]).T
        expected[:, 1:-1] += (percent[1:-1] - 25) * 0.4
        expected[:, -1] += 20.0
        self.assertArrayAlmostEqual(actual, expected)

    def test_masked_2d_multi_unequal(self):
        shape = (3, 10)
        data = ma.arange(np.prod(shape)).reshape(shape)
        weights = np.ones(shape)
        weights[0] = 3
        data[1] = ma.masked
        percent = np.array([30, 50, 75, 80])
        actual, weight_total = WPERCENTILE.aggregate(
            data, axis=0, percent=percent, weights=weights, returned=True
        )
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        expected = np.tile(np.arange(shape[-1]), percent.size)
        expected = expected.reshape(percent.size, shape[-1]).T
        expected[:, 1:] = 2.0 * (
            (0.875 - percent[1:] / 100.0) * data[0, np.newaxis].T
            + (percent[1:] / 100.0 - 0.375) * data[-1, np.newaxis].T
        )
        self.assertArrayAlmostEqual(actual, expected)
        self.assertTupleEqual(weight_total.shape, (shape[-1],))
        self.assertArrayEqual(weight_total, np.repeat(4, shape[-1]))

    def test_2d_multi_weight1d_unequal(self):
        shape = (3, 10)
        data = np.arange(np.prod(shape)).reshape(shape)
        weights1d = np.ones(shape[-1])
        weights1d[::3] = 3
        weights2d = np.broadcast_to(weights1d, shape)
        percent = np.array([30, 50, 75, 80])
        result_1d, wt_total_1d = WPERCENTILE.aggregate(
            data, axis=1, percent=percent, weights=weights1d, returned=True
        )
        result_2d, wt_total_2d = WPERCENTILE.aggregate(
            data, axis=1, percent=percent, weights=weights2d, returned=True
        )
        # Results should be the same whether we use 1d or 2d weights.
        self.assertArrayAllClose(result_1d, result_2d)


class Test_name(tests.IrisTest):
    def test(self):
        self.assertEqual(WPERCENTILE.name(), "weighted_percentile")


class Test_aggregate_shape(tests.IrisTest):
    def test_missing_mandatory_kwarg(self):
        emsg_pc = (
            "weighted_percentile aggregator requires .* keyword argument 'percent'"
        )
        emsg_wt = (
            "weighted_percentile aggregator requires .* keyword argument 'weights'"
        )
        with self.assertRaisesRegex(ValueError, emsg_pc):
            WPERCENTILE.aggregate_shape(weights=None)
        with self.assertRaisesRegex(ValueError, emsg_pc):
            kwargs = dict(weights=None)
            WPERCENTILE.aggregate_shape(**kwargs)
        with self.assertRaisesRegex(ValueError, emsg_pc):
            kwargs = dict(point=10)
            WPERCENTILE.aggregate_shape(**kwargs)
        with self.assertRaisesRegex(ValueError, emsg_wt):
            WPERCENTILE.aggregate_shape(percent=50)
        with self.assertRaisesRegex(ValueError, emsg_wt):
            kwargs = dict(percent=50)
            WPERCENTILE.aggregate_shape(**kwargs)
        with self.assertRaisesRegex(ValueError, emsg_wt):
            kwargs = dict(percent=50, weight=None)
            WPERCENTILE.aggregate_shape(**kwargs)

    def test_mandatory_kwarg_no_shape(self):
        kwargs = dict(percent=50, weights=None)
        self.assertTupleEqual(WPERCENTILE.aggregate_shape(**kwargs), ())
        kwargs = dict(percent=[50], weights=None)
        self.assertTupleEqual(WPERCENTILE.aggregate_shape(**kwargs), ())

    def test_mandatory_kwarg_shape(self):
        kwargs = dict(percent=(10, 20), weights=None)
        self.assertTupleEqual(WPERCENTILE.aggregate_shape(**kwargs), (2,))
        kwargs = dict(percent=range(13), weights=None)
        self.assertTupleEqual(WPERCENTILE.aggregate_shape(**kwargs), (13,))


class Test_cell_method(tests.IrisTest):
    def test(self):
        self.assertIsNone(WPERCENTILE.cell_method)


if __name__ == "__main__":
    tests.main()
