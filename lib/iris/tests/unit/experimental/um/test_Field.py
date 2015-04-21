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

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import mock
import numpy as np

from iris.experimental.um import Field


class Test_int_headers(tests.IrisTest):
    def test(self):
        field = Field(range(45), range(19), None)
        self.assertArrayEqual(field.int_headers, range(45))


class Test_real_headers(tests.IrisTest):
    def test(self):
        field = Field(range(45), range(19), None)
        self.assertArrayEqual(field.real_headers, range(19))


class Test___eq__(tests.IrisTest):
    def test_equal(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45), range(19), None)
        self.assertTrue(field1.__eq__(field2))

    def test_not_equal_ints(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45, 90), range(19), None)
        self.assertFalse(field1.__eq__(field2))

    def test_not_equal_reals(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45), range(19, 38), None)
        self.assertFalse(field1.__eq__(field2))

    def test_not_equal_data(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45), range(19), np.zeros(3))
        self.assertFalse(field1.__eq__(field2))

    def test_invalid(self):
        field1 = Field(range(45), range(19), None)
        self.assertIs(field1.__eq__('foo'), NotImplemented)


class Test___ne__(tests.IrisTest):
    def test_equal(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45), range(19), None)
        self.assertFalse(field1.__ne__(field2))

    def test_not_equal_ints(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45, 90), range(19), None)
        self.assertTrue(field1.__ne__(field2))

    def test_not_equal_reals(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45), range(19, 38), None)
        self.assertTrue(field1.__ne__(field2))

    def test_not_equal_data(self):
        field1 = Field(range(45), range(19), None)
        field2 = Field(range(45), range(19), np.zeros(3))
        self.assertTrue(field1.__ne__(field2))

    def test_invalid(self):
        field1 = Field(range(45), range(19), None)
        self.assertIs(field1.__ne__('foo'), NotImplemented)


class Test_num_values(tests.IrisTest):
    def test_64(self):
        field = Field(range(45), range(19), None)
        self.assertEqual(field.num_values(), 64)

    def test_128(self):
        field = Field(range(45), range(83), None)
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


class Test__raw_data_is_useable_with_lbpack(tests.IrisTest):
    def _check_formats(self, old_lbpack, new_lbpack, absent_provider=False):
        provider = mock.Mock(lookup_entry=mock.Mock(lbpack=old_lbpack))
        if absent_provider:
            # Replace the provider with a simple array.
            provider = np.zeros(2)
        field = Field(range(45), range(19), provider)
        return field._raw_data_is_useable_with_lbpack(new_lbpack)

    def test_okay_simple(self):
        self.assertTrue(self._check_formats(1234, 1234))

    def test_fail_simple(self):
        self.assertFalse(self._check_formats(1234, 1238))

    def test_fail_nodata(self):
        self.assertFalse(self._check_formats(1234, 1234, absent_provider=True))

    def test_fail_different_n1(self):
        self.assertFalse(self._check_formats(3000, 3001))

    def test_fail_different_n2(self):
        self.assertFalse(self._check_formats(3000, 3010))

    def test_fail_different_n3(self):
        self.assertFalse(self._check_formats(3000, 3100))

    def test_okay_compatible_n4(self):
        self.assertTrue(self._check_formats(2007, 3007))
        self.assertTrue(self._check_formats(7, 3007))

    def test_fail_incompatible_n4(self):
        self.assertFalse(self._check_formats(2007, 4007))

    def test_okay_same_unknown_n4(self):
        self.assertTrue(self._check_formats(8001, 8001))

    def test_fail_different_uppers(self):
        self.assertFalse(self._check_formats(101001111, 102001111))

    def test_okay_unknown_uppers(self):
        self.assertTrue(self._check_formats(123401111, 123401111))


class Test__check_valid_data_encoding(tests.IrisTest):
    def _check_formats(self, old_lbpack, new_lbpack, absent_provider=False):
        provider = mock.Mock(lookup_entry=mock.Mock(lbpack=old_lbpack))
        if absent_provider:
            # Replace the provider with a simple array.
            provider = np.zeros(2)
        field = Field(range(45), range(19), provider)
        field._check_valid_data_encoding(new_lbpack)

    def test_okay_unpacked_as_unpacked(self):
        self._check_formats(0, 0)

    def test_fail_unpacked_as_packed(self):
        re = 'lbpack=0 to .* lbpack=1 .* unsupported re-encoding'
        with self.assertRaisesRegexp(ValueError, re):
            self._check_formats(0, 1)

    def test_fail_array_as_packed(self):
        re = 'array .* lbpack=1 .* unsupported encoding'
        with self.assertRaisesRegexp(ValueError, re):
            self._check_formats(1, 1, absent_provider=True)

    def test_okay_unpacked_as_unpacked_equivalent_wordtypes(self):
        self._check_formats(2000, 3000)

    def test_fail_unpacked_as_unpacked_different_wordtypes(self):
        re = 'lbpack=2000 to .* lbpack=7000 .* unsupported re-encoding'
        with self.assertRaisesRegexp(ValueError, re):
            self._check_formats(2000, 7000)

    def test_okay_packed_as_packed(self):
        self._check_formats(1, 1)

    def test_fail_packed_as_packed_different(self):
        re = 'lbpack=1 to .* lbpack=2 .* unsupported re-encoding'
        with self.assertRaisesRegexp(ValueError, re):
            self._check_formats(1, 2)

    def test_okay_packed_as_packed_equivalent_wordtypes(self):
        self._check_formats(1, 3001)

    def test_fail_packed_as_packed_different_wordtypes(self):
        re = 'lbpack=2001 to .* lbpack=7001 .* unsupported re-encoding'
        with self.assertRaisesRegexp(ValueError, re):
            self._check_formats(2001, 7001)


if __name__ == '__main__':
    tests.main()
