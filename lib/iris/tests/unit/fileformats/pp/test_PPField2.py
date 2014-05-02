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
"""Unit tests for the `iris.fileformats.pp.PPField2` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.fileformats.pp as pp
from iris.fileformats.pp import PPField2, SplittableInt


class Test__init__(tests.IrisTest):
    def setUp(self):
        self.header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int)
        self.header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float)

    def test_no_headers(self):
        field = PPField2()
        self.assertIsNone(field._header_longs)
        self.assertIsNone(field._header_floats)
        self.assertIsNone(field.raw_lbpack)

    def test_lbpack_lookup(self):
        self.assertEqual(PPField2.HEADER_DICT['lbpack'], (20,))

    def test_raw_lbpack(self):
        raw_lbpack = 4321
        loc, = PPField2.HEADER_DICT['lbpack']
        self.header_longs[loc] = raw_lbpack
        field = PPField2(header_longs=self.header_longs)
        self.assertEqual(field.raw_lbpack, raw_lbpack)


class Test__getattr__(tests.IrisTest):
    def setUp(self):
        self.header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int)
        self.header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float)

    def test_attr_singular_long(self):
        lbrow = 1234
        loc, = PPField2.HEADER_DICT['lbrow']
        self.header_longs[loc] = lbrow
        field = PPField2(header_longs=self.header_longs)
        self.assertEqual(field.lbrow, lbrow)

    def test_attr_multi_long(self):
        lbuser = (100, 101, 102, 103, 104, 105, 106)
        loc = PPField2.HEADER_DICT['lbuser']
        self.header_longs[loc[0]:loc[-1] + 1] = lbuser
        field = PPField2(header_longs=self.header_longs)
        self.assertEqual(field.lbuser, lbuser)

    def test_attr_singular_float(self):
        bdatum = 1234
        loc, = PPField2.HEADER_DICT['bdatum']
        self.header_floats[loc - pp.NUM_LONG_HEADERS] = bdatum
        field = PPField2(header_floats=self.header_floats)
        self.assertEqual(field.bdatum, bdatum)

    def test_attr_multi_float(self):
        brsvd = (100, 101, 102, 103)
        loc = PPField2.HEADER_DICT['brsvd']
        start = loc[0] - pp.NUM_LONG_HEADERS
        stop = loc[-1] + 1 - pp.NUM_LONG_HEADERS
        self.header_floats[start:stop] = brsvd
        field = PPField2(header_floats=self.header_floats)
        self.assertEqual(field.brsvd, brsvd)

    def test_attr_special_lbpack(self):
        raw_lbpack = 4321
        loc, = PPField2.HEADER_DICT['lbpack']
        self.header_longs[loc] = raw_lbpack
        field = PPField2(header_longs=self.header_longs)
        result = field._lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_lbpack(self):
        raw_lbpack = 4321
        loc, = PPField2.HEADER_DICT['lbpack']
        self.header_longs[loc] = raw_lbpack
        field = PPField2(header_longs=self.header_longs)
        result = field.lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_raw_lbpack_assign(self):
        field = PPField2(header_longs=self.header_longs)
        self.assertEqual(field.raw_lbpack, 0)
        self.assertEqual(field.lbpack, 0)
        raw_lbpack = 4321
        field.lbpack = raw_lbpack
        self.assertEqual(field.raw_lbpack, raw_lbpack)
        self.assertNotIsInstance(field.raw_lbpack, SplittableInt)

    def test_attr_unknown(self):
        with self.assertRaises(AttributeError):
            PPField2().x


if __name__ == '__main__':
    tests.main()
