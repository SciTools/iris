# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.MEAN` aggregator."""

import numpy as np
import numpy.ma as ma
import pytest

from iris._lazy_data import as_concrete_data, as_lazy_data
from iris.analysis import MEAN
from iris.tests import _shared_utils


class Test_lazy_aggregate:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = ma.arange(12).reshape(3, 4)
        self.data.mask = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]]
        # --> fractions of masked-points in columns = [0, 1/3, 2/3, 1]
        self.array = as_lazy_data(self.data)
        self.axis = 0
        self.expected_masked = ma.mean(self.data, axis=self.axis)

    def test_mdtol_default(self):
        # Default operation is "mdtol=1" --> unmasked if *any* valid points.
        # --> output column masks = [0, 0, 0, 1]
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis)
        masked_result = as_concrete_data(agg)
        _shared_utils.assert_masked_array_almost_equal(
            masked_result, self.expected_masked
        )

    def test_mdtol_belowall(self):
        # Mdtol=0.25 --> masked columns = [0, 1, 1, 1]
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.25)
        masked_result = as_concrete_data(agg)
        expected_masked = self.expected_masked
        expected_masked.mask = [False, True, True, True]
        _shared_utils.assert_masked_array_almost_equal(masked_result, expected_masked)

    def test_mdtol_intermediate(self):
        # mdtol=0.5 --> masked columns = [0, 0, 1, 1]
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.5)
        masked_result = as_concrete_data(agg)
        expected_masked = self.expected_masked
        expected_masked.mask = [False, False, True, True]
        _shared_utils.assert_masked_array_almost_equal(masked_result, expected_masked)

    def test_mdtol_aboveall(self):
        # mdtol=0.75 --> masked columns = [0, 0, 0, 1]
        # In this case, effectively the same as mdtol=None.
        agg = MEAN.lazy_aggregate(self.array, axis=self.axis, mdtol=0.75)
        masked_result = as_concrete_data(agg)
        _shared_utils.assert_masked_array_almost_equal(
            masked_result, self.expected_masked
        )

    def test_multi_axis(self):
        data = np.arange(24.0).reshape((2, 3, 4))
        collapse_axes = (0, 2)
        lazy_data = as_lazy_data(data)
        agg = MEAN.lazy_aggregate(lazy_data, axis=collapse_axes)
        result = as_concrete_data(agg)
        expected = np.mean(data, axis=collapse_axes)
        _shared_utils.assert_array_all_close(result, expected)

    def test_last_axis(self):
        # From _setup:
        # self.data.mask = [[0, 0, 0, 1],
        #                   [0, 0, 1, 1],
        #                   [0, 1, 1, 1]]
        # --> fractions of masked-points in ROWS = [1/4, 1/2, 3/4]
        axis = -1
        agg = MEAN.lazy_aggregate(self.array, axis=axis, mdtol=0.51)
        expected_masked = ma.mean(self.data, axis=-1)
        expected_masked = np.ma.masked_array(expected_masked, [0, 0, 1])
        masked_result = as_concrete_data(agg)
        _shared_utils.assert_masked_array_almost_equal(masked_result, expected_masked)

    def test_all_axes_belowtol(self):
        agg = MEAN.lazy_aggregate(self.array, axis=None, mdtol=0.75)
        expected_masked = ma.mean(self.data)
        masked_result = as_concrete_data(agg)
        _shared_utils.assert_masked_array_almost_equal(masked_result, expected_masked)

    def test_all_axes_abovetol(self):
        agg = MEAN.lazy_aggregate(self.array, axis=None, mdtol=0.45)
        expected_masked = ma.masked_less([0.0], 1)
        masked_result = as_concrete_data(agg)
        _shared_utils.assert_masked_array_almost_equal(masked_result, expected_masked)


class Test_name:
    def test(self):
        assert MEAN.name() == "mean"


class Test_aggregate_shape:
    def test(self):
        shape = ()
        kwargs = dict()
        assert MEAN.aggregate_shape(**kwargs) == shape
        kwargs = dict(one=1, two=2)
        assert MEAN.aggregate_shape(**kwargs) == shape
