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
"""Unit tests for the `iris.fileformats.pp.PPField3` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.fileformats.pp as pp
from iris.fileformats.pp import PPField3, SplittableInt


class Test__init__(tests.IrisTest):
    def setUp(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float)
        self.header = list(header_longs) + list(header_floats)

    def test_no_headers(self):
        field = PPField3()
        self.assertIsNone(field._raw_header)
        self.assertIsNone(field.raw_lbtim)
        self.assertIsNone(field.raw_lbpack)

    def test_lbtim_lookup(self):
        self.assertEqual(PPField3.HEADER_DICT['lbtim'], (12,))

    def test_lbpack_lookup(self):
        self.assertEqual(PPField3.HEADER_DICT['lbpack'], (20,))

    def test_raw_lbtim(self):
        raw_lbtim = 4321
        loc, = PPField3.HEADER_DICT['lbtim']
        self.header[loc] = raw_lbtim
        field = PPField3(header=self.header)
        self.assertEqual(field.raw_lbtim, raw_lbtim)

    def test_raw_lbpack(self):
        raw_lbpack = 4321
        loc, = PPField3.HEADER_DICT['lbpack']
        self.header[loc] = raw_lbpack
        field = PPField3(header=self.header)
        self.assertEqual(field.raw_lbpack, raw_lbpack)


class Test__getattr__(tests.IrisTest):
    def setUp(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float)
        self.header = list(header_longs) + list(header_floats)

    def test_attr_singular_long(self):
        lbrow = 1234
        loc, = PPField3.HEADER_DICT['lbrow']
        self.header[loc] = lbrow
        field = PPField3(header=self.header)
        self.assertEqual(field.lbrow, lbrow)

    def test_attr_multi_long(self):
        lbuser = (100, 101, 102, 103, 104, 105, 106)
        loc = PPField3.HEADER_DICT['lbuser']
        self.header[loc[0]:loc[-1] + 1] = lbuser
        field = PPField3(header=self.header)
        self.assertEqual(field.lbuser, lbuser)

    def test_attr_singular_float(self):
        bdatum = 1234
        loc, = PPField3.HEADER_DICT['bdatum']
        self.header[loc] = bdatum
        field = PPField3(header=self.header)
        self.assertEqual(field.bdatum, bdatum)

    def test_attr_multi_float(self):
        brsvd = (100, 101, 102, 103)
        loc = PPField3.HEADER_DICT['brsvd']
        start = loc[0]
        stop = loc[-1] + 1
        self.header[start:stop] = brsvd
        field = PPField3(header=self.header)
        self.assertEqual(field.brsvd, brsvd)

    def test_attr_lbtim(self):
        raw_lbtim = 4321
        loc, = PPField3.HEADER_DICT['lbtim']
        self.header[loc] = raw_lbtim
        field = PPField3(header=self.header)
        result = field.lbtim
        self.assertEqual(result, raw_lbtim)
        self.assertIsInstance(result, SplittableInt)
        result = field._lbtim
        self.assertEqual(result, raw_lbtim)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_lbpack(self):
        raw_lbpack = 4321
        loc, = PPField3.HEADER_DICT['lbpack']
        self.header[loc] = raw_lbpack
        field = PPField3(header=self.header)
        result = field.lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)
        result = field._lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_raw_lbtim_assign(self):
        field = PPField3(header=self.header)
        self.assertEqual(field.raw_lbpack, 0)
        self.assertEqual(field.lbtim, 0)
        raw_lbtim = 4321
        field.lbtim = raw_lbtim
        self.assertEqual(field.raw_lbtim, raw_lbtim)
        self.assertNotIsInstance(field.raw_lbtim, SplittableInt)

    def test_attr_raw_lbpack_assign(self):
        field = PPField3(header=self.header)
        self.assertEqual(field.raw_lbpack, 0)
        self.assertEqual(field.lbpack, 0)
        raw_lbpack = 4321
        field.lbpack = raw_lbpack
        self.assertEqual(field.raw_lbpack, raw_lbpack)
        self.assertNotIsInstance(field.raw_lbpack, SplittableInt)

    def test_attr_unknown(self):
        with self.assertRaises(AttributeError):
            PPField3().x


if __name__ == '__main__':
    tests.main()
