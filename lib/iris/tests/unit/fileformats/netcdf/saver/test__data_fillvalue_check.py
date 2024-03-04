# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.fileformats.netcdf.saver._data_fillvalue_check`.

Note: now runs all testcases on both real + lazy data.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
import collections

import dask.array as da
import numpy as np

from iris.fileformats.netcdf.saver import _data_fillvalue_check


class Check__fillvalueandmasking:
    def _call_target(self, fill_value, keys, vals):
        data = np.zeros(20, dtype=np.float32)
        if any(np.ma.isMaskedArray(val) for val in vals):
            # N.B. array is masked if "vals" is, but has no masked points initially.
            data = np.ma.masked_array(data, mask=np.zeros_like(data))

        for key, val in zip(keys, vals):
            data[key] = val

        if hasattr(self.arraylib, "compute"):
            data = da.from_array(data, chunks=-1)

        results = _data_fillvalue_check(
            arraylib=self.arraylib, data=data, check_value=fill_value
        )

        if hasattr(results, "compute"):
            results = results.compute()

        # Return a named tuple, for named-property access to the 2 result values.
        result = collections.namedtuple("_", ["is_masked", "contains_value"])(
            *results
        )
        return result

    def test_no_fill_value_not_masked(self):
        # Test when the fill value is not present and the data is not masked
        keys = [slice(0, 10), slice(10, 15)]
        vals = [np.arange(10), np.arange(5)]
        fill_value = 16
        target = self._call_target(fill_value, keys, vals)
        self.assertFalse(target.contains_value)
        self.assertFalse(target.is_masked)

    def test_contains_fill_value_not_masked(self):
        # Test when the fill value is present and the data is not masked
        keys = [slice(0, 10), slice(10, 15)]
        vals = [np.arange(10), np.arange(5)]
        fill_value = 5
        target = self._call_target(fill_value, keys, vals)
        self.assertTrue(target.contains_value)
        self.assertFalse(target.is_masked)

    def test_no_fill_value_masked(self):
        # Test when the fill value is not present and the data is masked
        keys = [slice(0, 10), slice(10, 15)]
        vals = [np.arange(10), np.ma.masked_equal(np.arange(5), 3)]
        fill_value = 16
        target = self._call_target(fill_value, keys, vals)
        self.assertFalse(target.contains_value)
        self.assertTrue(target.is_masked)

    def test_contains_fill_value_masked(self):
        # Test when the fill value is present and the data is masked
        keys = [slice(0, 10), slice(10, 15)]
        vals = [np.arange(10), np.ma.masked_equal(np.arange(5), 3)]
        fill_value = 5
        target = self._call_target(fill_value, keys, vals)
        self.assertTrue(target.contains_value)
        self.assertTrue(target.is_masked)

    def test_fill_value_None(self):
        # Test when the fill value is None
        keys = [slice(0, 10), slice(10, 15)]
        vals = [np.arange(10), np.arange(5)]
        fill_value = None
        target = self._call_target(fill_value, keys, vals)
        self.assertFalse(target.contains_value)

    def test_contains_masked_fill_value(self):
        # Test when the fill value is present but masked the data is masked
        keys = [slice(0, 10), slice(10, 15)]
        vals = [np.arange(10), np.ma.masked_equal(np.arange(10, 15), 13)]
        fill_value = 13
        target = self._call_target(fill_value, keys, vals)
        self.assertFalse(target.contains_value)
        self.assertTrue(target.is_masked)


class Test__real(Check__fillvalueandmasking, tests.IrisTest):
    arraylib = np


class Test__lazy(Check__fillvalueandmasking, tests.IrisTest):
    arraylib = da
