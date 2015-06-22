# (C) British Crown Copyright 2014 - 2015, Met Office
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
Unit tests for :class:`iris.experimental.um.Field`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import range

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import mock
import numpy as np

from iris.experimental.um import Field


class Test_int_headers(tests.IrisTest):
    def test(self):
        field = Field(np.arange(45), list(range(19)), None)
        self.assertArrayEqual(field.int_headers, np.arange(45))


class Test_real_headers(tests.IrisTest):
    def test(self):
        field = Field(list(range(45)), np.arange(19), None)
        self.assertArrayEqual(field.real_headers, np.arange(19))


class Test___eq__(tests.IrisTest):
    def test_equal(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45), np.arange(19), None)
        self.assertTrue(field1.__eq__(field2))

    def test_not_equal_ints(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45, 90), np.arange(19), None)
        self.assertFalse(field1.__eq__(field2))

    def test_not_equal_reals(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45), np.arange(19, 38), None)
        self.assertFalse(field1.__eq__(field2))

    def test_not_equal_data(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45), np.arange(19), np.zeros(3))
        self.assertFalse(field1.__eq__(field2))

    def test_invalid(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        self.assertIs(field1.__eq__('foo'), NotImplemented)


class Test___ne__(tests.IrisTest):
    def test_equal(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45), np.arange(19), None)
        self.assertFalse(field1.__ne__(field2))

    def test_not_equal_ints(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45, 90), np.arange(19), None)
        self.assertTrue(field1.__ne__(field2))

    def test_not_equal_reals(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45), np.arange(19, 38), None)
        self.assertTrue(field1.__ne__(field2))

    def test_not_equal_data(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        field2 = Field(np.arange(45), np.arange(19), np.zeros(3))
        self.assertTrue(field1.__ne__(field2))

    def test_invalid(self):
        field1 = Field(list(range(45)), list(range(19)), None)
        self.assertIs(field1.__ne__('foo'), NotImplemented)


class Test_num_values(tests.IrisTest):
    def test_64(self):
        field = Field(list(range(45)), list(range(19)), None)
        self.assertEqual(field.num_values(), 64)

    def test_128(self):
        field = Field(list(range(45)), list(range(83)), None)
        self.assertEqual(field.num_values(), 128)


class Test_get_data(tests.IrisTest):
    def test_None(self):
        field = Field([], [], None)
        self.assertIsNone(field.get_data())

    def test_ndarray(self):
        data = np.arange(12).reshape(3, 4)
        field = Field([], [], data)
        self.assertIs(field.get_data(), data)

    def test_provider(self):
        provider = mock.Mock(read_data=lambda: mock.sentinel.DATA)
        field = Field([], [], provider)
        self.assertIs(field.get_data(), mock.sentinel.DATA)


class Test_set_data(tests.IrisTest):
    def test_None(self):
        data = np.arange(12).reshape(3, 4)
        field = Field([], [], data)
        field.set_data(None)
        self.assertIsNone(field.get_data())

    def test_ndarray(self):
        field = Field([], [], None)
        data = np.arange(12).reshape(3, 4)
        field.set_data(data)
        self.assertArrayEqual(field.get_data(), data)

    def test_provider(self):
        provider = mock.Mock(read_data=lambda: mock.sentinel.DATA)
        field = Field([], [], None)
        field.set_data(provider)
        self.assertIs(field.get_data(), mock.sentinel.DATA)


class Test__can_copy_deferred_data(tests.IrisTest):
    def _check_formats(self,
                       old_lbpack, new_lbpack,
                       old_bacc=-6, new_bacc=-6,
                       absent_provider=False):
        lookup_entry = mock.Mock(lbpack=old_lbpack, bacc=old_bacc)
        provider = mock.Mock(lookup_entry=lookup_entry)
        if absent_provider:
            # Replace the provider with a simple array.
            provider = np.zeros(2)
        field = Field(list(range(45)), list(range(19)), provider)
        return field._can_copy_deferred_data(new_lbpack, new_bacc)

    def test_okay_simple(self):
        self.assertTrue(self._check_formats(1234, 1234))

    def test_fail_different_lbpack(self):
        self.assertFalse(self._check_formats(1234, 1238))

    def test_fail_nodata(self):
        self.assertFalse(self._check_formats(1234, 1234, absent_provider=True))

    def test_fail_different_bacc(self):
        self.assertFalse(self._check_formats(1234, 1234, new_bacc=-8))


if __name__ == '__main__':
    tests.main()
