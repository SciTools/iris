# (C) British Crown Copyright 2013 - 2015, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :class:`iris.analysis.Aggregator` class instance."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.analysis import Aggregator
from iris.exceptions import LazyAggregatorError
from iris.tests import mock


class Test_aggregate(tests.IrisTest):
    # These unit tests don't call a data aggregation function, they call a
    # mocked one i.e. the return values of the mocked data aggregation
    # function don't matter, only how these are dealt with by the aggregate
    # method.
    def setUp(self):
        self.TEST = Aggregator('test', None)
        self.array = ma.array([[1, 2, 3],
                               [4, 5, 6]],
                              mask=[[False, True, False],
                                    [True, False, False]],
                              dtype=np.float64)
        self.expected_result_axis0 = ma.array([1, 2, 3], mask=None)
        self.expected_result_axis1 = ma.array([4, 5], mask=None)

    def test_masked_notol(self):
        # Providing masked array with no tolerance keyword (mdtol) provided.
        axis = 0
        mock_return = self.expected_result_axis0.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis)
            self.assertMaskedArrayEqual(result, self.expected_result_axis0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis)
            self.assertMaskedArrayEqual(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_masked_above_tol(self):
        # Providing masked array with a high tolerance (mdtol) provided.
        axis = 0
        mock_return = self.expected_result_axis0.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis, mdtol=.55)
            self.assertMaskedArrayEqual(result, self.expected_result_axis0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis, mdtol=.55)
            self.assertMaskedArrayEqual(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_masked_below_tol(self):
        # Providing masked array with a tolerance on missing values, low
        # enough to modify the resulting mask for axis 0.
        axis = 0
        result_axis_0 = self.expected_result_axis0.copy()
        result_axis_0.mask = np.array([True, True, False])
        mock_return = ma.array([1, 2, 3], mask=None)
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis, mdtol=.45)
            self.assertMaskedArrayAlmostEqual(result, result_axis_0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis, mdtol=.45)
            self.assertMaskedArrayEqual(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_masked_below_tol_alt(self):
        # Providing masked array with a tolerance on missing values, low
        # enough to modify the resulting mask for axis 1.
        axis = 1
        result_axis_1 = self.expected_result_axis1.copy()
        result_axis_1.mask = np.array([True, True])
        mock_return = self.expected_result_axis1.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis, mdtol=.1)
            self.assertMaskedArrayAlmostEqual(result, result_axis_1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_unmasked_with_mdtol(self):
        # Providing aggregator with an unmasked array and tolerance specified
        # for missing data - ensure that result is unaffected.
        data = self.array.data

        axis = 0
        mock_return = self.expected_result_axis0.data.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(data, axis, mdtol=0.5)
            self.assertArrayAlmostEqual(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.data.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(data, axis, mdtol=0.5)
            self.assertArrayAlmostEqual(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

    def test_unmasked(self):
        # Providing aggregator with an unmasked array and no additional keyword
        # arguments ensure that result is unaffected.
        data = self.array.data

        axis = 0
        mock_return = self.expected_result_axis0.data.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(data, axis)
            self.assertArrayAlmostEqual(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.data.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(data, axis)
            self.assertArrayAlmostEqual(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

    def test_returning_scalar_mdtol(self):
        # Test the case when the data aggregation function returns a scalar and
        # turns it into a masked array.
        axis = -1
        data = self.array.flatten()
        mock_return = 2
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(data, axis, mdtol=1)
            self.assertMaskedArrayEqual(result, ma.array(2, mask=False))
        mock_method.assert_called_once_with(data, axis=axis)

    def test_returning_scalar_mdtol_alt(self):
        # Test the case when the data aggregation function returns a scalar
        # with no tolerance for missing data values and turns it into a masked
        # array.
        axis = -1
        data = self.array.flatten()
        mock_return = 2
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(data, axis, mdtol=0)
            self.assertMaskedArrayEqual(result, ma.array(2, mask=True))
        mock_method.assert_called_once_with(data, axis=axis)

    def test_returning_non_masked_array_from_masked_array(self):
        # Providing a masked array, call_func returning a non-masked array,
        # resulting in a masked array output.
        axis = 0
        mock_return = self.expected_result_axis0.data.copy()
        result_axis_0 = ma.array(mock_return, mask=[True, True, False])
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis, mdtol=.45)
            self.assertMaskedArrayAlmostEqual(result, result_axis_0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.data.copy()
        with mock.patch.object(self.TEST, 'call_func',
                               return_value=mock_return) as mock_method:
            result = self.TEST.aggregate(self.array, axis, mdtol=.45)
            self.assertMaskedArrayEqual(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_kwarg_pass_through_no_kwargs(self):
        call_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        aggregator = Aggregator('', call_func)
        aggregator.aggregate(data, axis)
        call_func.assert_called_once_with(data, axis=axis)

    def test_kwarg_pass_through_call_kwargs(self):
        call_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        kwargs = dict(wibble='wobble', foo='bar')
        aggregator = Aggregator('', call_func)
        aggregator.aggregate(data, axis, **kwargs)
        call_func.assert_called_once_with(data, axis=axis, **kwargs)

    def test_kwarg_pass_through_init_kwargs(self):
        call_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        kwargs = dict(wibble='wobble', foo='bar')
        aggregator = Aggregator('', call_func, **kwargs)
        aggregator.aggregate(data, axis)
        call_func.assert_called_once_with(data, axis=axis, **kwargs)

    def test_kwarg_pass_through_combined_kwargs(self):
        call_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        init_kwargs = dict(wibble='wobble', var=1.0)
        call_kwargs = dict(foo='foo', var=0.5)
        aggregator = Aggregator('', call_func, **init_kwargs)
        aggregator.aggregate(data, axis, **call_kwargs)
        expected_kwargs = init_kwargs.copy()
        expected_kwargs.update(call_kwargs)
        call_func.assert_called_once_with(data, axis=axis, **expected_kwargs)

    def test_mdtol_intercept(self):
        call_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        aggregator = Aggregator('', call_func)
        aggregator.aggregate(data, axis, wibble='wobble', mdtol=0.8)
        call_func.assert_called_once_with(data, axis=axis, wibble='wobble')

    def test_no_lazy_func(self):
        dummy_agg = Aggregator('custom_op', lambda x: 1)
        expected = 'custom_op aggregator does not support lazy operation'
        with self.assertRaisesRegexp(LazyAggregatorError, expected):
            dummy_agg.lazy_aggregate(np.arange(10), axis=0)


class Test_update_metadata(tests.IrisTest):
    def test_no_units_change(self):
        # If the Aggregator has no units_func then the units should be
        # left unchanged.
        aggregator = Aggregator('', None)
        cube = mock.Mock(units=mock.sentinel.units)
        aggregator.update_metadata(cube, [])
        self.assertIs(cube.units, mock.sentinel.units)

    def test_units_change(self):
        # If the Aggregator has a units_func then the new units should
        # be defined by its return value.
        units_func = mock.Mock(return_value=mock.sentinel.new_units)
        aggregator = Aggregator('', None, units_func)
        cube = mock.Mock(units=mock.sentinel.units)
        aggregator.update_metadata(cube, [])
        units_func.assert_called_once_with(mock.sentinel.units)
        self.assertEqual(cube.units, mock.sentinel.new_units)


class Test_lazy_aggregate(tests.IrisTest):
    def test_kwarg_pass_through_no_kwargs(self):
        lazy_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        aggregator = Aggregator('', None, lazy_func=lazy_func)
        aggregator.lazy_aggregate(data, axis)
        lazy_func.assert_called_once_with(data, axis)

    def test_kwarg_pass_through_call_kwargs(self):
        lazy_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        kwargs = dict(wibble='wobble', foo='bar')
        aggregator = Aggregator('', None, lazy_func=lazy_func)
        aggregator.lazy_aggregate(data, axis, **kwargs)
        lazy_func.assert_called_once_with(data, axis, **kwargs)

    def test_kwarg_pass_through_init_kwargs(self):
        lazy_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        kwargs = dict(wibble='wobble', foo='bar')
        aggregator = Aggregator('', None, lazy_func=lazy_func, **kwargs)
        aggregator.lazy_aggregate(data, axis)
        lazy_func.assert_called_once_with(data, axis, **kwargs)

    def test_kwarg_pass_through_combined_kwargs(self):
        lazy_func = mock.Mock()
        data = mock.sentinel.data
        axis = mock.sentinel.axis
        init_kwargs = dict(wibble='wobble', var=1.0)
        call_kwargs = dict(foo='foo', var=0.5)
        aggregator = Aggregator('', None, lazy_func=lazy_func, **init_kwargs)
        aggregator.lazy_aggregate(data, axis, **call_kwargs)
        expected_kwargs = init_kwargs.copy()
        expected_kwargs.update(call_kwargs)
        lazy_func.assert_called_once_with(data, axis, **expected_kwargs)


if __name__ == "__main__":
    tests.main()
