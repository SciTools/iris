# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.MEDIAN` aggregator."""

import numpy as np
import numpy.ma as ma
import pytest

from iris._lazy_data import (
    as_concrete_data,
    as_lazy_data,
    is_lazy_data,
    is_lazy_masked_data,
)
from iris.analysis import MEDIAN
from iris.tests import _shared_utils


def _get_data(lazy=False, masked=False):
    data = np.arange(16).reshape((4, 4))
    if masked:
        mask = np.eye(4)
        data = ma.masked_array(data=data, mask=mask)
    if lazy:
        data = as_lazy_data(data)
    return data


class Test_basics:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = _get_data()

    def test_name(self):
        assert MEDIAN.name() == "median"

    def test_collapse(self):
        data = MEDIAN.aggregate(self.data, axis=(0, 1))
        _shared_utils.assert_array_equal(data, [7.5])


class Test_masked:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = _get_data(masked=True)

    def test_output_is_masked(self):
        result = MEDIAN.aggregate(self.data, axis=1)
        assert ma.isMaskedArray(result)

    def test_median_is_mask_aware(self):
        # the median computed along one axis differs if the array is masked
        axis = 1
        result = MEDIAN.aggregate(self.data, axis=axis)
        data_no_mask = _get_data()
        result_no_mask = MEDIAN.aggregate(data_no_mask, axis=axis)

        # In lieu of `assert_array_NOT_all_close`
        with pytest.raises(AssertionError, match="ARRAY CHECK FAILED"):
            _shared_utils.assert_array_all_close(result, result_no_mask)


class Test_lazy:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = _get_data(lazy=True)

    def test_output_is_lazy(self):
        result = MEDIAN.lazy_aggregate(self.data, axis=(0, 1))
        assert is_lazy_data(result)

    def test_shape(self):
        result = MEDIAN.lazy_aggregate(self.data, axis=1)
        assert result.shape == (4,)

    def test_result_values(self):
        axis = 1
        result = MEDIAN.lazy_aggregate(self.data, axis=axis)
        expected = np.median(as_concrete_data(self.data), axis=axis)
        _shared_utils.assert_array_almost_equal(result, expected)


class Test_lazy_masked:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = _get_data(lazy=True, masked=True)

    def test_output_is_lazy_and_masked(self):
        result = MEDIAN.lazy_aggregate(self.data, axis=1)
        assert is_lazy_masked_data(result)

    def test_result_values(self):
        axis = 1
        result = MEDIAN.lazy_aggregate(self.data, axis=axis)
        expected = ma.median(as_concrete_data(self.data), axis=axis)
        _shared_utils.assert_array_almost_equal(result, expected)
