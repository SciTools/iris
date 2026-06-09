# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.analysis.Aggregator` class instance."""

import numpy as np
import numpy.ma as ma
import pytest

from iris.analysis import Aggregator
from iris.cube import Cube
from iris.exceptions import LazyAggregatorError
from iris.tests import _shared_utils


class Test_aggregate:
    # These unit tests don't call a data aggregation function, they call a
    # mocked one i.e. the return values of the mocked data aggregation
    # function don't matter, only how these are dealt with by the aggregate
    # method.
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.TEST = Aggregator("test", None)
        self.array = ma.array(
            [[1, 2, 3], [4, 5, 6]],
            mask=[[False, True, False], [True, False, False]],
            dtype=np.float64,
        )
        self.expected_result_axis0 = ma.array([1, 2, 3], mask=None)
        self.expected_result_axis1 = ma.array([4, 5], mask=None)

    def test_masked_notol(self, mocker):
        # Providing masked array with no tolerance keyword (mdtol) provided.
        axis = 0
        mock_return = self.expected_result_axis0.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis)
        _shared_utils.assert_masked_array_equal(result, self.expected_result_axis0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis)
        _shared_utils.assert_masked_array_equal(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_masked_above_tol(self, mocker):
        # Providing masked array with a high tolerance (mdtol) provided.
        axis = 0
        mock_return = self.expected_result_axis0.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis, mdtol=0.55)
        _shared_utils.assert_masked_array_equal(result, self.expected_result_axis0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis, mdtol=0.55)
        _shared_utils.assert_masked_array_equal(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_masked_below_tol(self, mocker):
        # Providing masked array with a tolerance on missing values, low
        # enough to modify the resulting mask for axis 0.
        axis = 0
        result_axis_0 = self.expected_result_axis0.copy()
        result_axis_0.mask = np.array([True, True, False])
        mock_return = ma.array([1, 2, 3], mask=None)
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis, mdtol=0.45)
        _shared_utils.assert_masked_array_almost_equal(result, result_axis_0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis, mdtol=0.45)
        _shared_utils.assert_masked_array_equal(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_masked_below_tol_alt(self, mocker):
        # Providing masked array with a tolerance on missing values, low
        # enough to modify the resulting mask for axis 1.
        axis = 1
        result_axis_1 = self.expected_result_axis1.copy()
        result_axis_1.mask = np.array([True, True])
        mock_return = self.expected_result_axis1.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis, mdtol=0.1)
        _shared_utils.assert_masked_array_almost_equal(result, result_axis_1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_unmasked_with_mdtol(self, mocker):
        # Providing aggregator with an unmasked array and tolerance specified
        # for missing data - ensure that result is unaffected.
        data = self.array.data

        axis = 0
        mock_return = self.expected_result_axis0.data.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(data, axis, mdtol=0.5)
        _shared_utils.assert_array_almost_equal(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.data.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(data, axis, mdtol=0.5)
        _shared_utils.assert_array_almost_equal(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

    def test_unmasked(self, mocker):
        # Providing aggregator with an unmasked array and no additional keyword
        # arguments ensure that result is unaffected.
        data = self.array.data

        axis = 0
        mock_return = self.expected_result_axis0.data.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(data, axis)
        _shared_utils.assert_array_almost_equal(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.data.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(data, axis)
        _shared_utils.assert_array_almost_equal(result, mock_return.copy())
        mock_method.assert_called_once_with(data, axis=axis)

    def test_allmasked_1_d_with_mdtol(self, mocker):
        data = ma.masked_all((3,))
        axis = 0
        mdtol = 0.5
        mock_return = ma.masked
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(data, axis, mdtol=mdtol)

        assert result is mock_return
        mock_method.assert_called_once_with(data, axis=axis)

    def test_returning_scalar_mdtol(self, mocker):
        # Test the case when the data aggregation function returns a scalar and
        # turns it into a masked array.
        axis = -1
        data = self.array.flatten()
        mock_return = 2
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(data, axis, mdtol=1)
        _shared_utils.assert_masked_array_equal(result, ma.array(2, mask=False))
        mock_method.assert_called_once_with(data, axis=axis)

    def test_returning_scalar_mdtol_alt(self, mocker):
        # Test the case when the data aggregation function returns a scalar
        # with no tolerance for missing data values and turns it into a masked
        # array.
        axis = -1
        data = self.array.flatten()
        mock_return = 2
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(data, axis, mdtol=0)
        _shared_utils.assert_masked_array_equal(result, ma.array(2, mask=True))
        mock_method.assert_called_once_with(data, axis=axis)

    def test_returning_non_masked_array_from_masked_array(self, mocker):
        # Providing a masked array, call_func returning a non-masked array,
        # resulting in a masked array output.
        axis = 0
        mock_return = self.expected_result_axis0.data.copy()
        result_axis_0 = ma.array(mock_return, mask=[True, True, False])
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis, mdtol=0.45)
        _shared_utils.assert_masked_array_almost_equal(result, result_axis_0)
        mock_method.assert_called_once_with(self.array, axis=axis)

        axis = 1
        mock_return = self.expected_result_axis1.data.copy()
        mock_method = mocker.patch.object(
            self.TEST, "call_func", return_value=mock_return
        )
        result = self.TEST.aggregate(self.array, axis, mdtol=0.45)
        _shared_utils.assert_masked_array_equal(result, self.expected_result_axis1)
        mock_method.assert_called_once_with(self.array, axis=axis)

    def test_kwarg_pass_through_no_kwargs(self, mocker):
        call_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        aggregator = Aggregator("", call_func)
        aggregator.aggregate(data, axis)
        call_func.assert_called_once_with(data, axis=axis)

    def test_kwarg_pass_through_call_kwargs(self, mocker):
        call_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        kwargs = dict(wibble="wobble", foo="bar")
        aggregator = Aggregator("", call_func)
        aggregator.aggregate(data, axis, **kwargs)
        call_func.assert_called_once_with(data, axis=axis, **kwargs)

    def test_kwarg_pass_through_init_kwargs(self, mocker):
        call_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        kwargs = dict(wibble="wobble", foo="bar")
        aggregator = Aggregator("", call_func, **kwargs)
        aggregator.aggregate(data, axis)
        call_func.assert_called_once_with(data, axis=axis, **kwargs)

    def test_kwarg_pass_through_combined_kwargs(self, mocker):
        call_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        init_kwargs = dict(wibble="wobble", var=1.0)
        call_kwargs = dict(foo="foo", var=0.5)
        aggregator = Aggregator("", call_func, **init_kwargs)
        aggregator.aggregate(data, axis, **call_kwargs)
        expected_kwargs = init_kwargs.copy()
        expected_kwargs.update(call_kwargs)
        call_func.assert_called_once_with(data, axis=axis, **expected_kwargs)

    def test_mdtol_intercept(self, mocker):
        call_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        aggregator = Aggregator("", call_func)
        aggregator.aggregate(data, axis, wibble="wobble", mdtol=0.8)
        call_func.assert_called_once_with(data, axis=axis, wibble="wobble")

    def test_no_lazy_func(self):
        dummy_agg = Aggregator("custom_op", lambda x: 1)
        expected = "custom_op aggregator does not support lazy operation"
        with pytest.raises(LazyAggregatorError, match=expected):
            dummy_agg.lazy_aggregate(np.arange(10), axis=0)


class Test_update_metadata:
    def test_no_units_change(self, mocker):
        # If the Aggregator has no units_func then the units should be
        # left unchanged.
        aggregator = Aggregator("", None)
        cube = mocker.Mock(units=mocker.sentinel.units)
        aggregator.update_metadata(cube, [])
        assert cube.units is mocker.sentinel.units

    def test_units_change(self, mocker):
        # If the Aggregator has a units_func then the new units should
        # be defined by its return value.
        units_func = mocker.Mock(return_value=mocker.sentinel.new_units)
        aggregator = Aggregator("", None, units_func)
        cube = mocker.Mock(units=mocker.sentinel.units)
        aggregator.update_metadata(cube, [], kw1=1, kw2=2)
        units_func.assert_called_once_with(mocker.sentinel.units, kw1=1, kw2=2)
        assert cube.units == mocker.sentinel.new_units

    def test_units_func_no_kwargs(self):
        # To ensure backwards-compatibility, Aggregator also supports
        # units_func that accept the single argument `units`
        def units_func(units):
            return units**2

        aggregator = Aggregator("", None, units_func)
        cube = Cube(0, units="s")
        aggregator.update_metadata(cube, [], kw1=1, kw2=2)
        assert cube.units == "s2"

    def test_units_func_kwargs(self):
        def units_func(units, **kwargs):
            return units**2

        aggregator = Aggregator("", None, units_func)
        cube = Cube(0, units="s")
        aggregator.update_metadata(cube, [], kw1=1, kw2=2)
        assert cube.units == "s2"


class Test_lazy_aggregate:
    def test_kwarg_pass_through_no_kwargs(self, mocker):
        lazy_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        aggregator = Aggregator("", None, lazy_func=lazy_func)
        aggregator.lazy_aggregate(data, axis)
        lazy_func.assert_called_once_with(data, axis=axis)

    def test_kwarg_pass_through_call_kwargs(self, mocker):
        lazy_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        kwargs = dict(wibble="wobble", foo="bar")
        aggregator = Aggregator("", None, lazy_func=lazy_func)
        aggregator.lazy_aggregate(data, axis, **kwargs)
        lazy_func.assert_called_once_with(data, axis=axis, **kwargs)

    def test_kwarg_pass_through_init_kwargs(self, mocker):
        lazy_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        kwargs = dict(wibble="wobble", foo="bar")
        aggregator = Aggregator("", None, lazy_func=lazy_func, **kwargs)
        aggregator.lazy_aggregate(data, axis)
        lazy_func.assert_called_once_with(data, axis=axis, **kwargs)

    def test_kwarg_pass_through_combined_kwargs(self, mocker):
        lazy_func = mocker.Mock()
        data = mocker.sentinel.data
        axis = mocker.sentinel.axis
        init_kwargs = dict(wibble="wobble", var=1.0)
        call_kwargs = dict(foo="foo", var=0.5)
        aggregator = Aggregator("", None, lazy_func=lazy_func, **init_kwargs)
        aggregator.lazy_aggregate(data, axis, **call_kwargs)
        expected_kwargs = init_kwargs.copy()
        expected_kwargs.update(call_kwargs)
        lazy_func.assert_called_once_with(data, axis=axis, **expected_kwargs)
