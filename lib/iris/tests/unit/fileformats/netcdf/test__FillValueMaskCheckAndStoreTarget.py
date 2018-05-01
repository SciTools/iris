# (C) British Crown Copyright 2017 - 2018, Met Office
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
"""
Unit tests for the `iris.fileformats.netcdf._FillValueMaskCheckAndStoreTarget`
class.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.fileformats.netcdf import _FillValueMaskCheckAndStoreTarget
from iris.tests import mock


class Test__FillValueMaskCheckAndStoreTarget(tests.IrisTest):
    def _call_target(self, fill_value, keys, vals):
        inner_target = mock.MagicMock()
        target = _FillValueMaskCheckAndStoreTarget(inner_target,
                                                   fill_value=fill_value)

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
