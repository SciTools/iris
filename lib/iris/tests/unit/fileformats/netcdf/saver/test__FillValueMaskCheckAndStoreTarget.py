# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `iris.fileformats.netcdf._FillValueMaskCheckAndStoreTarget`
class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.fileformats.netcdf.saver import _FillValueMaskCheckAndStoreTarget


class Test__FillValueMaskCheckAndStoreTarget(tests.IrisTest):
    def _call_target(self, fill_value, keys, vals):
        inner_target = mock.MagicMock()
        target = _FillValueMaskCheckAndStoreTarget(
            inner_target, fill_value=fill_value
        )

        for key, val in zip(keys, vals):
            target[key] = val

        calls = [mock.call(key, val) for key, val in zip(keys, vals)]
        inner_target.__setitem__.assert_has_calls(calls)

        return target

    def test___setitem__(self):
        self._call_target(None, [1], [2])

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
