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
"""Unit tests for the `cf_units.Unit` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

import cf_units
from cf_units import Unit


class Test___init__(tests.IrisTest):

    def test_capitalised_calendar(self):
        calendar = 'GrEgoRian'
        expected = cf_units.CALENDAR_GREGORIAN
        u = Unit('hours since 1970-01-01 00:00:00', calendar=calendar)
        self.assertEqual(u.calendar, expected)

    def test_not_basestring_calendar(self):
        with self.assertRaises(TypeError):
            u = Unit('hours since 1970-01-01 00:00:00', calendar=5)


class Test_convert__calendar(tests.IrisTest):

    class MyStr(str):
        pass

    def test_gregorian_calendar_conversion_setup(self):
        # Reproduces a situation where a unit's gregorian calendar would not
        # match (using the `is` operator) to the literal string 'gregorian',
        # causing an `is not` test to return a false negative.
        cal_str = cf_units.CALENDAR_GREGORIAN
        calendar = self.MyStr(cal_str)
        self.assertIsNot(calendar, cal_str)
        u1 = Unit('hours since 1970-01-01 00:00:00', calendar=calendar)
        u2 = Unit('hours since 1969-11-30 00:00:00', calendar=calendar)
        u1point = np.array([8.], dtype=np.float32)
        expected = np.array([776.], dtype=np.float32)
        result = u1.convert(u1point, u2)
        return expected, result

    def test_gregorian_calendar_conversion_array(self):
        expected, result = self.test_gregorian_calendar_conversion_setup()
        self.assertArrayEqual(expected, result)

    def test_gregorian_calendar_conversion_dtype(self):
        expected, result = self.test_gregorian_calendar_conversion_setup()
        self.assertEqual(expected.dtype, result.dtype)

    def test_gregorian_calendar_conversion_shape(self):
        expected, result = self.test_gregorian_calendar_conversion_setup()
        self.assertEqual(expected.shape, result.shape)

    def test_non_gregorian_calendar_conversion_dtype(self):
        data = np.arange(4, dtype=np.float32)
        u1 = Unit('hours since 2000-01-01 00:00:00', calendar='360_day')
        u2 = Unit('hours since 2000-01-02 00:00:00', calendar='360_day')
        result = u1.convert(data, u2)
        self.assertEqual(result.dtype, np.float32)


class Test_convert__endianness_time(tests.IrisTest):
    # Test the behaviour of converting time units of differing
    # dtype endianness.

    def setUp(self):
        self.time1_array = np.array([31.5, 32.5, 33.5])
        self.time2_array = np.array([0.5, 1.5, 2.5])
        self.time1_unit = cf_units.Unit('days since 1970-01-01 00:00:00',
                                        calendar=cf_units.CALENDAR_STANDARD)
        self.time2_unit = cf_units.Unit('days since 1970-02-01 00:00:00',
                                        calendar=cf_units.CALENDAR_STANDARD)

    def test_no_endian(self):
        dtype = 'f8'
        result = self.time1_unit.convert(self.time1_array.astype(dtype),
                                         self.time2_unit)
        self.assertArrayAlmostEqual(result, self.time2_array)

    def test_little_endian(self):
        dtype = '<f8'
        result = self.time1_unit.convert(self.time1_array.astype(dtype),
                                         self.time2_unit)
        self.assertArrayAlmostEqual(result, self.time2_array)

    def test_big_endian(self):
        dtype = '>f8'
        result = self.time1_unit.convert(self.time1_array.astype(dtype),
                                         self.time2_unit)
        self.assertArrayAlmostEqual(result, self.time2_array)


class Test_convert__endianness_deg_to_rad(tests.IrisTest):
    # Test the behaviour of converting radial units of differing
    # dtype endianness.

    def setUp(self):
        self.degs_array = np.array([356.7, 356.8, 356.9])
        self.rads_array = np.array([6.22558944, 6.22733477, 6.2290801])
        self.deg = cf_units.Unit('degrees')
        self.rad = cf_units.Unit('radians')

    def test_no_endian(self):
        dtype = 'f8'
        result = self.deg.convert(self.degs_array.astype(dtype), self.rad)
        self.assertArrayAlmostEqual(result, self.rads_array)

    def test_little_endian(self):
        dtype = '<f8'
        result = self.deg.convert(self.degs_array.astype(dtype), self.rad)
        self.assertArrayAlmostEqual(result, self.rads_array)

    def test_big_endian(self):
        dtype = '>f8'
        result = self.deg.convert(self.degs_array.astype(dtype), self.rad)
        self.assertArrayAlmostEqual(result, self.rads_array)


class Test_convert__endianness_degC_to_kelvin(tests.IrisTest):
    # Test the behaviour of converting temperature units of differing
    # dtype endianness.

    def setUp(self):
        self.k_array = np.array([356.7, 356.8, 356.9])
        self.degc_array = np.array([83.55, 83.65, 83.75])
        self.degc = cf_units.Unit('degC')
        self.k = cf_units.Unit('K')

    def test_no_endian(self):
        dtype = 'f8'
        result = self.degc.convert(self.degc_array.astype(dtype), self.k)
        self.assertArrayAlmostEqual(result, self.k_array)

    def test_little_endian(self):
        dtype = '<f8'
        result = self.degc.convert(self.degc_array.astype(dtype), self.k)
        self.assertArrayAlmostEqual(result, self.k_array)

    def test_big_endian(self):
        dtype = '>f8'
        result = self.degc.convert(self.degc_array.astype(dtype), self.k)
        self.assertArrayAlmostEqual(result, self.k_array)


class Test_convert__result_ctype(tests.IrisTest):
    # Test the output ctype of converting an Iris unit.

    def setUp(self):
        self.initial_dtype = np.float32
        self.degs_array = np.array([356.7, 356.8, 356.9],
                                   dtype=self.initial_dtype)
        self.deg = cf_units.Unit('degrees')
        self.rad = cf_units.Unit('radians')

    def test_default(self):
        # The dtype of a float array should be unchanged.
        result = self.deg.convert(self.degs_array, self.rad)
        self.assertEqual(result.dtype, self.initial_dtype)

    def test_integer_ctype_default(self):
        # The ctype of an int array should be cast to the default ctype.
        result = self.deg.convert(self.degs_array.astype(np.int32), self.rad)
        self.assertEqual(result.dtype, cf_units.FLOAT64)

    def test_integer_ctype_specified(self):
        # The ctype of an int array should be cast to the specified ctype if
        # supplied.
        expected_dtype = cf_units.FLOAT32
        result = self.deg.convert(self.degs_array.astype(np.int32), self.rad,
                                  ctype=expected_dtype)
        self.assertEqual(result.dtype, expected_dtype)


class Test_convert__masked_array(tests.IrisTest):
    # Test converting an Iris unit with masked data.

    def setUp(self):
        self.deg = cf_units.Unit('degrees')
        self.rad = cf_units.Unit('radians')
        self.degs_array = np.ma.array(np.array([356.7, 356.8, 356.9],
                                               dtype=np.float32),
                                      mask=np.array([0, 1, 0], dtype=bool))
        self.rads_array = np.ma.array(np.array([6.22558944,
                                                6.22733477,
                                                6.2290801],
                                               dtype=np.float32),
                                      mask=np.array([0, 1, 0], dtype=bool))

    def test_no_type_conversion(self):
        result = self.deg.convert(self.degs_array, self.rad,
                                  ctype=cf_units.FLOAT32)
        self.assertArrayAlmostEqual(self.rads_array, result)

    def test_type_conversion(self):
        result = self.deg.convert(self.degs_array, self.rad,
                                  ctype=cf_units.FLOAT64)
        self.assertArrayAlmostEqual(self.rads_array, result)


if __name__ == '__main__':
    tests.main()
