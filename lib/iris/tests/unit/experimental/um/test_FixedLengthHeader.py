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
Unit tests for :class:`iris.experimental.um.FixedLengthHeader`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.experimental.um import FixedLengthHeader


class Test_empty(tests.IrisTest):
    def check(self, dtype, word_size=None):
        if word_size is None:
            header = FixedLengthHeader.empty()
        else:
            header = FixedLengthHeader.empty(word_size)
        self.assertArrayEqual(header.raw, [-32768] * 256)
        self.assertEqual(header.raw.dtype, dtype)

    def test_default(self):
        self.check('>i8')

    def test_explicit_64_bit(self):
        self.check('>i8', 8)

    def test_explicit_32_bit(self):
        self.check('>i4', 4)


class Test_from_file(tests.IrisTest):
    def check(self, src_dtype, word_size=None):
        data = (np.arange(1000) * 10).astype(src_dtype)
        with self.temp_filename() as filename:
            data.tofile(filename)
            with open(filename, 'rb') as source:
                if word_size is None:
                    header = FixedLengthHeader.from_file(source)
                else:
                    header = FixedLengthHeader.from_file(source, word_size)
        self.assertArrayEqual(header.raw, np.arange(256) * 10)

    def test_default(self):
        self.check('>i8')

    def test_explicit_64_bit(self):
        self.check('>i8', 8)

    def test_explicit_32_bit(self):
        self.check('>i4', 4)


class Test___init__(tests.IrisTest):
    def test_invalid_length(self):
        with self.assertRaisesRegexp(ValueError, 'Incorrect number of words'):
            FixedLengthHeader(range(15))


def make_header():
    return FixedLengthHeader((np.arange(256) + 1) * 10)


class Test_data_set_format_version(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.data_set_format_version, 10)


class Test_sub_model(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.sub_model, 20)


class Test_total_prognostic_fields(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.total_prognostic_fields, 1530)


class Test_integer_constants_start(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.integer_constants_start, 1000)


class Test_integer_constants_shape(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.integer_constants_shape, (1010,))


class Test_row_dependent_constants_shape(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.row_dependent_constants_shape, (1160, 1170))


class Test_data_shape(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.data_shape, (1610,))


class Test_max_length(tests.IrisTest):
    def test(self):
        header = make_header()
        self.assertEqual(header.max_length, (1620,))


if __name__ == '__main__':
    tests.main()
