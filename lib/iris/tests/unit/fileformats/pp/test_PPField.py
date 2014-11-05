# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the `iris.fileformats.pp.PPField` class."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.fileformats.pp as pp
from iris.fileformats.pp import PPField
from iris.fileformats.pp import SplittableInt

# The PPField class is abstract, so to test we define a minimal,
# concrete subclass with the `t1` and `t2` properties.
#
# NB. We define dummy header items to allow us to zero the unused header
# items when written to disk and get consistent results.


DUMMY_HEADER = [('dummy1', (0, 13)),
                ('lbtim', (12,)),
                ('lblrec', (14,)),
                ('dummy2', (15, 18)),
                ('lbrow', (17,)),
                ('lbext',  (19,)),
                ('lbpack', (20,)),
                ('dummy3', (21, 37)),
                ('lbuser', (38, 39, 40, 41, 42, 43, 44,)),
                ('brsvd', (45, 46, 47, 48)),
                ('bdatum', (49,)),
                ('dummy4', (45, 63)),
                ]


class TestPPField(PPField):

    HEADER_DEFN = DUMMY_HEADER
    HEADER_DICT = dict(DUMMY_HEADER)

    @property
    def t1(self):
        return netcdftime.datetime(2013, 10, 14, 10, 4)

    @property
    def t2(self):
        return netcdftime.datetime(2013, 10, 14, 10, 5)


class Test_save(tests.IrisTest):
    def test_float64(self):
        # Tests down-casting of >f8 data to >f4.

        def field_checksum(data):
            field = TestPPField()
            field.dummy1 = 0
            field.dummy2 = 0
            field.dummy3 = 0
            field.dummy4 = 0
            field.lbtim = 0
            field.lblrec = 0
            field.lbrow = 0
            field.lbext = 0
            field.lbpack = 0
            field.lbuser = 0
            field.brsvd = 0
            field.bdatum = 0
            field.data = data
            with self.temp_filename('.pp') as temp_filename:
                with open(temp_filename, 'wb') as pp_file:
                    field.save(pp_file)
                checksum = self.file_checksum(temp_filename)
            return checksum

        data_64 = np.linspace(0, 1, num=10, endpoint=False).reshape(2, 5)
        checksum_32 = field_checksum(data_64.astype('>f4'))
        with mock.patch('warnings.warn') as warn:
            checksum_64 = field_checksum(data_64.astype('>f8'))

        self.assertEquals(checksum_32, checksum_64)
        warn.assert_called()


class Test_calendar(tests.IrisTest):
    def test_greg(self):
        field = TestPPField()
        field.lbtim = SplittableInt(1, {'ia': 2, 'ib': 1, 'ic': 0})
        self.assertEqual(field.calendar, 'gregorian')

    def test_360(self):
        field = TestPPField()
        field.lbtim = SplittableInt(2, {'ia': 2, 'ib': 1, 'ic': 0})
        self.assertEqual(field.calendar, '360_day')

    def test_365(self):
        field = TestPPField()
        field.lbtim = SplittableInt(4, {'ia': 2, 'ib': 1, 'ic': 0})
        self.assertEqual(field.calendar, '365_day')


class Test__init__(tests.IrisTest):
    def setUp(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float)
        self.header = list(header_longs) + list(header_floats)

    def test_no_headers(self):
        field = TestPPField()
        self.assertIsNone(field._raw_header)
        self.assertIsNone(field.raw_lbtim)
        self.assertIsNone(field.raw_lbpack)

    def test_lbtim_lookup(self):
        self.assertEqual(TestPPField.HEADER_DICT['lbtim'], (12,))

    def test_lbpack_lookup(self):
        self.assertEqual(TestPPField.HEADER_DICT['lbpack'], (20,))

    def test_raw_lbtim(self):
        raw_lbtim = 4321
        loc, = TestPPField.HEADER_DICT['lbtim']
        self.header[loc] = raw_lbtim
        field = TestPPField(header=self.header)
        self.assertEqual(field.raw_lbtim, raw_lbtim)

    def test_raw_lbpack(self):
        raw_lbpack = 4321
        loc, = TestPPField.HEADER_DICT['lbpack']
        self.header[loc] = raw_lbpack
        field = TestPPField(header=self.header)
        self.assertEqual(field.raw_lbpack, raw_lbpack)


class Test__getattr__(tests.IrisTest):
    def setUp(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float)
        self.header = list(header_longs) + list(header_floats)

    def test_attr_singular_long(self):
        lbrow = 1234
        loc, = TestPPField.HEADER_DICT['lbrow']
        self.header[loc] = lbrow
        field = TestPPField(header=self.header)
        self.assertEqual(field.lbrow, lbrow)

    def test_attr_multi_long(self):
        lbuser = (100, 101, 102, 103, 104, 105, 106)
        loc = TestPPField.HEADER_DICT['lbuser']
        self.header[loc[0]:loc[-1] + 1] = lbuser
        field = TestPPField(header=self.header)
        self.assertEqual(field.lbuser, lbuser)

    def test_attr_singular_float(self):
        bdatum = 1234
        loc, = TestPPField.HEADER_DICT['bdatum']
        self.header[loc] = bdatum
        field = TestPPField(header=self.header)
        self.assertEqual(field.bdatum, bdatum)

    def test_attr_multi_float(self):
        brsvd = (100, 101, 102, 103)
        loc = TestPPField.HEADER_DICT['brsvd']
        start = loc[0]
        stop = loc[-1] + 1
        self.header[start:stop] = brsvd
        field = TestPPField(header=self.header)
        self.assertEqual(field.brsvd, brsvd)

    def test_attr_lbtim(self):
        raw_lbtim = 4321
        loc, = TestPPField.HEADER_DICT['lbtim']
        self.header[loc] = raw_lbtim
        field = TestPPField(header=self.header)
        result = field.lbtim
        self.assertEqual(result, raw_lbtim)
        self.assertIsInstance(result, SplittableInt)
        result = field._lbtim
        self.assertEqual(result, raw_lbtim)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_lbpack(self):
        raw_lbpack = 4321
        loc, = TestPPField.HEADER_DICT['lbpack']
        self.header[loc] = raw_lbpack
        field = TestPPField(header=self.header)
        result = field.lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)
        result = field._lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_raw_lbtim_assign(self):
        field = TestPPField(header=self.header)
        self.assertEqual(field.raw_lbpack, 0)
        self.assertEqual(field.lbtim, 0)
        raw_lbtim = 4321
        field.lbtim = raw_lbtim
        self.assertEqual(field.raw_lbtim, raw_lbtim)
        self.assertNotIsInstance(field.raw_lbtim, SplittableInt)

    def test_attr_raw_lbpack_assign(self):
        field = TestPPField(header=self.header)
        self.assertEqual(field.raw_lbpack, 0)
        self.assertEqual(field.lbpack, 0)
        raw_lbpack = 4321
        field.lbpack = raw_lbpack
        self.assertEqual(field.raw_lbpack, raw_lbpack)
        self.assertNotIsInstance(field.raw_lbpack, SplittableInt)

    def test_attr_unknown(self):
        with self.assertRaises(AttributeError):
            TestPPField().x


if __name__ == "__main__":
    tests.main()
