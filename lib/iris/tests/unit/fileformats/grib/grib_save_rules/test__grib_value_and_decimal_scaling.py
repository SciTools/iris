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
"""
Unit tests for `iris.fileformats.grib.grib_save_rules._decimal_scaled_int`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests


import numpy as np

from iris.fileformats.grib.grib_save_rules import _decimal_scaled_int


class Test__integers(tests.IrisTest):
    _grib_max_positive_int = 2 ** 63 - 1

    def test_zero(self):
        test_value = 0
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(value, 0)
        self.assertEqual(scaling, 0)

    def test_nonzero(self):
        test_value = 12321
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(value, test_value)
        self.assertEqual(scaling, 0)

    def test_negative(self):
        test_value = -1
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(value, test_value)
        self.assertEqual(scaling, 0)

    def test_not_too_large(self):
        test_value = self._grib_max_positive_int / 2
        value, scaling = _decimal_scaled_int(test_value)
        # NOTE: this testcase *not* exact, as value != int(round(value)).
        self.assertLessEqual(abs(value - test_value), 1)
        self.assertEqual(scaling, 0)

    def test_too_large(self):
        test_value = self._grib_max_positive_int * 3
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(scaling, -1)
        # NOTE: accuracy here is ~20 digit, *not* limited by floating point.
        self.assertLessEqual(abs(value - round(test_value) / 10), 1)

    def test_large_negative(self):
        test_value = -200 * self._grib_max_positive_int
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(scaling, -3)
        self.assertLessEqual(abs(value - round(test_value / 1000)), 1)


class Test__floats(tests.IrisTest):
    def test_zero(self):
        test_value = 0.0
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(value, 0)
        self.assertEqual(scaling, 0)

    def test_nonzero(self):
        test_value = 123.123
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(scaling, 3)
        self.assertEqual(value, 123123)

    def test_negative(self):
        test_value = -123.5
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(scaling, 1)
        self.assertEqual(value, -1235)

    def test_inexact(self):
        test_value = 1.0 / 9
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(scaling, 16)
        self.assertEqual(value, 1111111111111111)

    def test_float32(self):
        test_value = np.array(1.0 / 9, dtype=np.float32)
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(scaling, 8)
        self.assertEqual(value, 11111111)

    def test_float64(self):
        test_value = np.array(1.0 / 9, dtype=np.float64)
        value, scaling = _decimal_scaled_int(test_value)
        self.assertEqual(scaling, 16)
        self.assertEqual(value, 1111111111111111)


if __name__ == "__main__":
    tests.main()
