# (C) British Crown Copyright 2014, Met Office
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


class Test_read_data(tests.IrisTest):
    def test_None(self):
        field = Field([], [], None)
        self.assertIsNone(field.read_data())

    def test_ndarray(self):
        data = np.arange(12).reshape(3, 4)
        field = Field([], [], data)
        self.assertIs(field.read_data(), data)

    def test_provider(self):
        provider = mock.Mock(read_data=lambda field: mock.sentinel.DATA)
        field = Field([], [], provider)
        self.assertIs(field.read_data(), mock.sentinel.DATA)


if __name__ == '__main__':
    tests.main()
