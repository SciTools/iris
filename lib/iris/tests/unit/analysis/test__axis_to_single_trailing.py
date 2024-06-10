# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis._axis_to_single_trailing` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import dask.array as da
import numpy as np

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
from iris.analysis import _axis_to_single_trailing


class TestInputReshape(tests.IrisTest):
    """Tests to make sure correct array is passed into stat function."""

    def setUp(self):
        self.stat_func = mock.Mock()

    def check_input(self, data, axis, expected):
        """Given data and axis passed to the wrapped function, check that expected
        array is passed to the inner function.

        """
        wrapped_stat_func = _axis_to_single_trailing(self.stat_func)
        wrapped_stat_func(data, axis=axis)
        # Can't use Mock.assert_called_with because array equality is ambiguous
        # get hold of the first arg instead.
        self.assertArrayEqual(self.stat_func.call_args.args[0], expected)

    def test_1d_input(self):
        # Trailing axis chosen, so array should be unchanged.
        data = np.arange(5)
        axis = 0
        self.check_input(data, axis, data)

    def test_2d_input_trailing(self):
        # Trailing axis chosen, so array should be unchanged.
        data = np.arange(6).reshape(2, 3)
        axis = 1
        self.stat_func.return_value = np.empty(2)
        self.check_input(data, axis, data)

    def test_2d_input_transpose(self):
        # Leading axis chosen, so array should be transposed.
        data = np.arange(6).reshape(2, 3)
        axis = 0
        self.stat_func.return_value = np.empty(3)
        self.check_input(data, axis, data.T)

    def test_3d_input_middle(self):
        # Middle axis is chosen, should be moved to end. Other dims should be
        # flattened.
        data = np.arange(24).reshape(2, 3, 4)
        axis = 1
        self.stat_func.return_value = np.empty(8)
        expected = np.moveaxis(data, 1, 2).reshape(8, 3)
        self.check_input(data, axis, expected)

    def test_3d_input_leading_multiple(self):
        # First 2 axis chosen, should be flattened and moved to end.
        data = np.arange(24).reshape(2, 3, 4)
        axis = (0, 1)
        self.stat_func.return_value = np.empty(4)
        expected = np.moveaxis(data, 2, 0).reshape(4, 6)
        self.check_input(data, axis, expected)

    def test_4d_first_and_last(self):
        data = np.arange(120).reshape(2, 3, 4, 5)
        axis = (0, -1)
        self.stat_func.return_value = np.empty(12)
        expected = np.moveaxis(data, 0, 2).reshape(12, 10)
        self.check_input(data, axis, expected)

    def test_3d_input_leading_multiple_lazy(self):
        # First 2 axis chosen, should be flattened and moved to end.  Lazy data
        # should be preserved.
        data = np.arange(24).reshape(2, 3, 4)
        lazy_data = as_lazy_data(data)
        axis = (0, 1)
        self.stat_func.return_value = np.empty(4)
        expected = np.moveaxis(data, 2, 0).reshape(4, 6)

        wrapped_stat_func = _axis_to_single_trailing(self.stat_func)
        wrapped_stat_func(lazy_data, axis=axis)
        self.assertTrue(is_lazy_data(self.stat_func.call_args.args[0]))
        self.assertArrayEqual(
            as_concrete_data(self.stat_func.call_args.args[0]), expected
        )


class TestOutputReshape(tests.IrisTest):
    """Tests to make sure array from stat function is handled correctly."""

    def setUp(self):
        self.stat_func = mock.Mock()

    def test_1d_input_1d_output(self):
        # If array is fully aggregated, result should be same as returned by stat
        # function.
        data = np.arange(3)
        self.stat_func.return_value = np.arange(2)
        wrapped_stat_func = _axis_to_single_trailing(self.stat_func)
        result = wrapped_stat_func(data, axis=0)
        self.assertArrayEqual(result, self.stat_func.return_value)

    def test_3d_input_middle_single_stat(self):
        # result shape should match non-aggregated input dims.
        data = np.empty((2, 3, 4))
        axis = 1
        self.stat_func.return_value = np.arange(8)
        expected = np.arange(8).reshape(2, 4)
        wrapped_stat_func = _axis_to_single_trailing(self.stat_func)
        result = wrapped_stat_func(data, axis=axis)
        self.assertArrayEqual(result, expected)

    def test_3d_input_middle_single_stat_lazy(self):
        # result shape should match non-aggregated input dims.  Lazy data should
        # be preserved.
        data = np.empty((2, 3, 4))
        axis = 1
        self.stat_func.return_value = da.arange(8)
        expected = np.arange(8).reshape(2, 4)
        wrapped_stat_func = _axis_to_single_trailing(self.stat_func)
        result = wrapped_stat_func(data, axis=axis)
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(as_concrete_data(result), expected)

    def test_3d_input_middle_multiple_stat(self):
        # result shape should match non-aggregated input dims, plus trailing dim
        # with size determined by the stat function.
        data = np.empty((2, 3, 4))
        axis = 1
        self.stat_func.return_value = np.arange(8 * 5).reshape(8, 5)
        expected = np.arange(40).reshape(2, 4, 5)
        wrapped_stat_func = _axis_to_single_trailing(self.stat_func)
        result = wrapped_stat_func(data, axis=axis)
        self.assertArrayEqual(result, expected)


if __name__ == "__main__":
    tests.main()
