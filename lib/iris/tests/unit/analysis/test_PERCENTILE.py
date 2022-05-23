# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :data:`iris.analysis.PERCENTILE` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
from iris.analysis import PERCENTILE


class AggregateMixin:
    """
    Percentile aggregation tests for both numpy and scipy methods within lazy
    and real percentile aggregation.

    """

    def check_percentile_calc(
        self, data, axis, percent, expected, approx=False, **kwargs
    ):
        if self.lazy:
            data = as_lazy_data(data)

        expected = ma.array(expected)

        actual = self.agg_method(
            data,
            axis=axis,
            percent=percent,
            fast_percentile_method=self.fast,
            **kwargs,
        )

        self.assertTupleEqual(actual.shape, expected.shape)
        is_lazy = is_lazy_data(actual)

        if self.lazy:
            self.assertTrue(is_lazy)
            actual = as_concrete_data(actual)
        else:
            self.assertFalse(is_lazy)

        if approx:
            self.assertMaskedArrayAlmostEqual(actual, expected)
        else:
            self.assertMaskedArrayEqual(actual, expected)

    def test_1d_single(self):
        data = np.arange(11)
        axis = 0
        percent = 50
        expected = 5
        self.check_percentile_calc(data, axis, percent, expected)

    def test_1d_multi(self):
        data = np.arange(11)
        percent = np.array([20, 50, 90])
        axis = 0
        expected = [2, 5, 9]
        self.check_percentile_calc(data, axis, percent, expected)

    def test_2d_single(self):
        shape = (2, 11)
        data = np.arange(np.prod(shape)).reshape(shape)
        axis = 0
        percent = 50
        expected = np.arange(shape[-1]) + 5.5
        self.check_percentile_calc(data, axis, percent, expected)

    def test_2d_multi(self):
        shape = (2, 10)
        data = np.arange(np.prod(shape)).reshape(shape)
        axis = 0
        percent = np.array([10, 50, 90, 100])
        expected = np.tile(np.arange(shape[-1]), percent.size)
        expected = expected.reshape(percent.size, shape[-1]).T + 1
        expected = expected + (percent / 10 - 1)
        self.check_percentile_calc(data, axis, percent, expected, approx=True)


class ScipyAggregateMixin:
    """
    Tests for calculations specific to the default (scipy) function.  Includes
    tests on masked data and tests to verify that the function is called with
    the expected keywords.  Needs to be used with AggregateMixin, as some of
    these tests re-use its method.

    """

    def test_masked_1d_single(self):
        data = ma.arange(11)
        data[3:7] = ma.masked
        axis = 0
        percent = 50
        expected = 7
        self.check_percentile_calc(data, axis, percent, expected)

    def test_masked_1d_multi(self):
        data = ma.arange(11)
        data[3:9] = ma.masked
        percent = np.array([25, 50, 75])
        axis = 0
        expected = [1, 2, 9]
        self.check_percentile_calc(data, axis, percent, expected)

    def test_masked_2d_single(self):
        shape = (2, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data[1, 1::2] = ma.masked
        axis = 0
        percent = 50
        # data has only one value for each column being aggregated, so result
        # should be that value.
        expected = np.empty(shape[-1:])
        expected[1::2] = data[0, 1::2]
        expected[::2] = data[1, ::2]
        self.check_percentile_calc(data, axis, percent, expected)

    def test_masked_2d_multi(self):
        shape = (3, 10)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[1, ::2] = ma.masked
        percent = np.array([10, 50, 70, 80])
        axis = 0
        mdtol = 0.1

        # First column is just 0 and 20.  Percentiles of these can be calculated as
        # linear interpolation.
        expected = percent / 100 * 20
        # Other columns are first column plus column number.
        expected = ma.array(
            np.broadcast_to(expected, (shape[-1], percent.size))
            + np.arange(shape[-1])[:, np.newaxis]
        )
        expected[::2] = ma.masked

        self.check_percentile_calc(
            data, axis, percent, expected, mdtol=mdtol, approx=True
        )

    @mock.patch("scipy.stats.mstats.mquantiles", return_value=[2, 4])
    def test_default_kwargs_passed(self, mocked_mquantiles):
        data = np.arange(5)
        percent = [42, 75]
        axis = 0
        if self.lazy:
            data = as_lazy_data(data)

        self.agg_method(data, axis=axis, percent=percent)

        # Trigger calculation for lazy case.
        as_concrete_data(data)
        for key in ["alphap", "betap"]:
            self.assertEqual(mocked_mquantiles.call_args.kwargs[key], 1)

    @mock.patch("scipy.stats.mstats.mquantiles")
    def test_chosen_kwargs_passed(self, mocked_mquantiles):
        data = np.arange(5)
        percent = [42, 75]
        axis = 0
        if self.lazy:
            data = as_lazy_data(data)

        self.agg_method(
            data, axis=axis, percent=percent, alphap=0.6, betap=0.5
        )

        # Trigger calculation for lazy case.
        as_concrete_data(data)
        for key, val in zip(["alphap", "betap"], [0.6, 0.5]):
            self.assertEqual(mocked_mquantiles.call_args.kwargs[key], val)


class Test_aggregate(tests.IrisTest, AggregateMixin, ScipyAggregateMixin):
    """Tests for standard aggregation method on real data."""

    def setUp(self):
        self.fast = False
        self.lazy = False
        self.agg_method = PERCENTILE.aggregate

    def test_missing_mandatory_kwarg(self):
        emsg = "percentile aggregator requires .* keyword argument 'percent'"
        with self.assertRaisesRegex(ValueError, emsg):
            PERCENTILE.aggregate("dummy", axis=0)


class Test_fast_aggregate(tests.IrisTest, AggregateMixin):
    """Tests for fast percentile method on real data."""

    def setUp(self):
        self.fast = True
        self.lazy = False
        self.agg_method = PERCENTILE.aggregate

    def test_masked(self):
        # Using (3,11) because np.percentile returns a masked array anyway with
        # (2, 11)
        shape = (3, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        emsg = (
            "Cannot use fast np.percentile method with masked array unless "
            "mdtol is 0."
        )
        with self.assertRaisesRegex(TypeError, emsg):
            PERCENTILE.aggregate(
                data, axis=0, percent=50, fast_percentile_method=True
            )

    def test_masked_mdtol_0(self):
        # Using (3,11) because np.percentile returns a masked array anyway with
        # (2, 11)
        shape = (3, 11)
        axis = 0
        percent = 50
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        expected = ma.arange(shape[-1]) + 11
        expected[::2] = ma.masked
        self.check_percentile_calc(data, axis, percent, expected, mdtol=0)

    @mock.patch("numpy.percentile")
    def test_numpy_percentile_called(self, mocked_percentile):
        # Basic check that numpy.percentile is called.
        data = np.arange(5)
        self.agg_method(data, axis=0, percent=42, fast_percentile_method=True)
        mocked_percentile.assert_called_once()


class MultiAxisMixin:
    """
    Tests for axis passed as a tuple.  Only relevant for lazy aggregation since
    axis is always specified as int for real aggregation.

    """

    def test_multi_axis(self):
        data = np.arange(24).reshape((2, 3, 4))
        collapse_axes = (0, 2)
        lazy_data = as_lazy_data(data)
        percent = 30
        actual = PERCENTILE.lazy_aggregate(
            lazy_data,
            axis=collapse_axes,
            percent=percent,
            fast_percentile_method=self.fast,
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
            lazy_data,
            axis=collapse_axes,
            percent=percent,
            fast_percentile_method=self.fast,
        )
        self.assertTrue(is_lazy_data(actual))
        result = as_concrete_data(actual)
        self.assertTupleEqual(result.shape, (3, 5))
        for num, sub_result in enumerate(result):
            # results should be the same as percentiles calculated from slices.
            self.assertArrayAlmostEqual(
                sub_result, np.percentile(data[:, num, :], percent)
            )


class Test_lazy_fast_aggregate(tests.IrisTest, AggregateMixin, MultiAxisMixin):
    """Tests for fast aggregation on lazy data."""

    def setUp(self):
        self.fast = True
        self.lazy = True
        self.agg_method = PERCENTILE.lazy_aggregate

    def test_masked(self):
        shape = (2, 11)
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data = as_lazy_data(data)
        actual = PERCENTILE.lazy_aggregate(
            data, axis=0, percent=50, fast_percentile_method=True
        )
        emsg = (
            "Cannot use fast np.percentile method with masked array unless "
            "mdtol is 0."
        )
        with self.assertRaisesRegex(TypeError, emsg):
            as_concrete_data(actual)

    def test_masked_mdtol_0(self):
        # Using (3,11) because np.percentile returns a masked array anyway with
        # (2, 11)
        shape = (3, 11)
        axis = 0
        percent = 50
        data = ma.arange(np.prod(shape)).reshape(shape)
        data[0, ::2] = ma.masked
        data = as_lazy_data(data)
        expected = ma.arange(shape[-1]) + 11
        expected[::2] = ma.masked
        self.check_percentile_calc(data, axis, percent, expected, mdtol=0)

    @mock.patch("numpy.percentile", return_value=np.array([2, 4]))
    def test_numpy_percentile_called(self, mocked_percentile):
        # Basic check that numpy.percentile is called.
        data = da.arange(5)
        result = self.agg_method(
            data, axis=0, percent=[42, 75], fast_percentile_method=True
        )

        self.assertTrue(is_lazy_data(result))
        as_concrete_data(result)
        mocked_percentile.assert_called()


class Test_lazy_aggregate(
    tests.IrisTest, AggregateMixin, ScipyAggregateMixin, MultiAxisMixin
):
    """Tests for standard aggregation on lazy data."""

    def setUp(self):
        self.fast = False
        self.lazy = True
        self.agg_method = PERCENTILE.lazy_aggregate


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
