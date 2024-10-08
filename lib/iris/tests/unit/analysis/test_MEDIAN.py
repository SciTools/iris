# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :data:`iris.analysis.MEDIAN` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris._lazy_data import (
    as_concrete_data,
    as_lazy_data,
    is_lazy_data,
    is_lazy_masked_data,
)
from iris.analysis import MEDIAN


def _get_data(lazy=False, masked=False):
    data = np.arange(16).reshape((4, 4))
    if masked:
        mask = np.eye(4)
        data = ma.masked_array(data=data, mask=mask)
    if lazy:
        data = as_lazy_data(data)
    return data


class Test_basics(tests.IrisTest):
    def setUp(self):
        self.data = _get_data()

    def test_name(self):
        self.assertEqual(MEDIAN.name(), "median")

    def test_collapse(self):
        data = MEDIAN.aggregate(self.data, axis=(0, 1))
        self.assertArrayEqual(data, [7.5])


class Test_masked(tests.IrisTest):
    def setUp(self):
        self.data = _get_data(masked=True)

    def test_output_is_masked(self):
        result = MEDIAN.aggregate(self.data, axis=1)
        self.assertTrue(ma.isMaskedArray(result))

    def test_median_is_mask_aware(self):
        # the median computed along one axis differs if the array is masked
        axis = 1
        result = MEDIAN.aggregate(self.data, axis=axis)
        data_no_mask = _get_data()
        result_no_mask = MEDIAN.aggregate(data_no_mask, axis=axis)
        self.assertFalse(np.allclose(result, result_no_mask))


class Test_lazy(tests.IrisTest):
    def setUp(self):
        self.data = _get_data(lazy=True)

    def test_output_is_lazy(self):
        result = MEDIAN.lazy_aggregate(self.data, axis=(0, 1))
        self.assertTrue(is_lazy_data(result))

    def test_shape(self):
        result = MEDIAN.lazy_aggregate(self.data, axis=1)
        self.assertTupleEqual(result.shape, (4,))

    def test_result_values(self):
        axis = 1
        result = MEDIAN.lazy_aggregate(self.data, axis=axis)
        expected = np.median(as_concrete_data(self.data), axis=axis)
        self.assertArrayAlmostEqual(result, expected)


class Test_lazy_masked(tests.IrisTest):
    def setUp(self):
        self.data = _get_data(lazy=True, masked=True)

    def test_output_is_lazy_and_masked(self):
        result = MEDIAN.lazy_aggregate(self.data, axis=1)
        self.assertTrue(is_lazy_masked_data(result))

    def test_result_values(self):
        axis = 1
        result = MEDIAN.lazy_aggregate(self.data, axis=axis)
        expected = ma.median(as_concrete_data(self.data), axis=axis)
        self.assertArrayAlmostEqual(result, expected)


if __name__ == "__main__":
    tests.main()
