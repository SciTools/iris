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

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
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


class LazyMixin:
    """Tests for both numpy and scipy methods within lazy percentile aggregation."""

    def test_1d_single(self):
        data = as_lazy_data(np.arange(11))
        actual = PERCENTILE.lazy_aggregate(
            data, axis=0, percent=50, fast_percentile_method=self.fast
        )
        expected = 5
        self.assertTupleEqual(actual.shape, ())
        self.assertTrue(is_lazy_data(actual))
        self.assertEqual(as_concrete_data(actual), expected)

    def test_1d_multi(self):
        data = as_lazy_data(np.arange(11))
        percent = np.array([20, 50, 90])
        actual = PERCENTILE.lazy_aggregate(
            data, axis=0, percent=percent, fast_percentile_method=self.fast
        )
        expected = [2, 5, 9]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertTrue(is_lazy_data(actual))
        self.assertArrayEqual(as_concrete_data(actual), expected)

    def test_2d_single(self):
        shape = (2, 11)
        data = as_lazy_data(np.arange(np.prod(shape)).reshape(shape))
        actual = PERCENTILE.lazy_aggregate(
            data, axis=0, percent=50, fast_percentile_method=self.fast
        )
        self.assertTupleEqual(actual.shape, shape[-1:])
        self.assertTrue(is_lazy_data(actual))
        expected = np.arange(shape[-1]) + 5.5
        self.assertArrayEqual(as_concrete_data(actual), expected)

    def test_2d_multi(self):
        shape = (2, 10)
        data = as_lazy_data(np.arange(np.prod(shape)).reshape(shape))
        percent = np.array([10, 50, 90, 100])
        actual = PERCENTILE.lazy_aggregate(
            data, axis=0, percent=percent, fast_percentile_method=self.fast
        )
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        self.assertTrue(is_lazy_data(actual))
        expected = np.tile(np.arange(shape[-1]), percent.size)
        expected = expected.reshape(percent.size, shape[-1]).T + 1
        expected = expected + (percent / 10 - 1)
        self.assertArrayAlmostEqual(as_concrete_data(actual), expected)


class Test_lazy_fast_aggregate(tests.IrisTest, LazyMixin):
    def setUp(self):
        self.fast = True

    def test_masked(self):
        shape = (2, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data = as_lazy_data(data)
        actual = PERCENTILE.lazy_aggregate(
            data, axis=0, percent=50, fast_percentile_method=True
        )
        emsg = "Cannot use fast np.percentile method with masked array."
        with self.assertRaisesRegex(TypeError, emsg):
            as_concrete_data(actual)

    def test_multi_axis(self):
        data = np.arange(24).reshape((2, 3, 4))
        collapse_axes = (0, 2)
        lazy_data = as_lazy_data(data)
        percent = 30
        actual = PERCENTILE.lazy_aggregate(
            lazy_data, axis=collapse_axes, percent=percent
        )
        self.assertTrue(is_lazy_data(actual))
        result = as_concrete_data(actual)
        self.assertTupleEqual(result.shape, (3,))
        for num, sub_result in enumerate(result):
            # results should be the same as percentiles calculated from slices.
            self.assertArrayAlmostEqual(
                sub_result, np.percentile(data[:, num, :], percent)
            )

    def test_multi_axis_multi_percent(self):
        data = np.arange(24).reshape((2, 3, 4))
        collapse_axes = (0, 2)
        lazy_data = as_lazy_data(data)
        percent = [20, 30, 50, 70, 80]
        actual = PERCENTILE.lazy_aggregate(
            lazy_data, axis=collapse_axes, percent=percent
        )
        self.assertTrue(is_lazy_data(actual))
        result = as_concrete_data(actual)
        self.assertTupleEqual(result.shape, (3, 5))
        for num, sub_result in enumerate(result):
            # results should be the same as percentiles calculated from slices.
            self.assertArrayAlmostEqual(
                sub_result, np.percentile(data[:, num, :], percent)
            )


class Test_lazy_aggregate(tests.IrisTest, LazyMixin):
    def setUp(self):
        self.fast = False

    def test_masked_1d_single(self):
        data = ma.arange(11)
        data[3:7] = ma.masked
        data = as_lazy_data(data)
        actual = PERCENTILE.lazy_aggregate(data, axis=0, percent=50)
        expected = 7
        self.assertTupleEqual(actual.shape, ())
        self.assertTrue(is_lazy_data(actual))
        self.assertEqual(as_concrete_data(actual), expected)

    def test_masked_1d_multi(self):
        data = ma.arange(11)
        data[3:9] = ma.masked
        data = as_lazy_data(data)
        percent = np.array([25, 50, 75])
        actual = PERCENTILE.lazy_aggregate(data, axis=0, percent=percent)
        expected = [1, 2, 9]
        self.assertTupleEqual(actual.shape, percent.shape)
        self.assertTrue(is_lazy_data(actual))
        self.assertArrayEqual(as_concrete_data(actual), expected)

    def test_masked_2d_single(self):
        shape = (2, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data[1, 1::2] = ma.masked
        data = as_lazy_data(data)
        actual = PERCENTILE.lazy_aggregate(data, axis=0, percent=50)
        self.assertTupleEqual(actual.shape, shape[-1:])
        self.assertTrue(is_lazy_data(actual))
        # data has only one value for each column being aggregated, so result
        # should be that value.
        expected = np.empty(shape[-1:])
        expected[1::2] = data[0, 1::2]
        expected[::2] = data[1, ::2]
        self.assertArrayEqual(as_concrete_data(actual), expected)

    def test_masked_2d_multi(self):
        shape = (3, 10)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[1] = ma.masked
        data = as_lazy_data(data)
        percent = np.array([10, 50, 70, 80])
        actual = PERCENTILE.lazy_aggregate(
            data, axis=0, percent=percent, mdtol=0.1
        )
        self.assertTupleEqual(actual.shape, (shape[-1], percent.size))
        self.assertTrue(is_lazy_data(actual))
        # First column is just 0 and 20.  Percentiles of these can be calculated as
        # linear interpolation.
        expected = percent / 100 * 20
        # Other columns are first column plus column number.
        expected = (
            np.broadcast_to(expected, (shape[-1], percent.size))
            + np.arange(shape[-1])[:, np.newaxis]
        )
        self.assertArrayAlmostEqual(as_concrete_data(actual), expected)


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
