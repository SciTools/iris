# (C) British Crown Copyright 2017, Met Office
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
Test :func:`iris._lazy data.array_nans_to_masked` function.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests


import numpy as np
import numpy.ma as ma

from iris._lazy_data import array_nans_to_masked


class Test(tests.IrisTest):
    def test_pass_thru_masked_array(self):
        array = ma.arange(10, dtype=np.float)
        result = array_nans_to_masked(array)
        self.assertTrue(isinstance(result, ma.MaskedArray))
        self.assertIs(result, array)

    def test_pass_thru_integral_array(self):
        array = np.arange(10)
        result = array_nans_to_masked(array)
        self.assertFalse(isinstance(result, ma.MaskedArray))
        self.assertIs(result, array)

    def test_no_nans(self):
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = array_nans_to_masked(array)
        self.assertFalse(isinstance(result, ma.MaskedArray))
        self.assertIs(result, array)

    def test_nans(self):
        array = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = array_nans_to_masked(array)
        self.assertTrue(isinstance(result, ma.MaskedArray))
        self.assertArrayEqual(result.mask, [[False, True], [False, False]])
        sentinal = 666.0
        result[0, 1] = sentinal
        self.assertArrayEqual(result.data, [[1.0, sentinal], [3.0, 4.0]])
        # Check that fill value is the "standard" one for the type.
        expected = ma.masked_array(array[0], dtype=array.dtype)
        self.assertEqual(result.fill_value, expected.fill_value)

    def test_fill_value_default(self):
        array = np.array([1.0, np.nan])
        result = array_nans_to_masked(array)
        self.assertTrue(isinstance(result, ma.MaskedArray))
        # Check that fill value is the "standard" one for the type.
        expected = ma.masked_array(array[0], dtype=array.dtype)
        self.assertEqual(result.fill_value, expected.fill_value)

    def test_fill_value_set(self):
        array = np.array([1.0, np.nan])
        fill_value = 666.0
        result = array_nans_to_masked(array, fill_value=fill_value)
        self.assertTrue(isinstance(result, ma.MaskedArray))
        # Check fill value is the specified custom value.
        self.assertEqual(result.fill_value, fill_value)

    def test_filled(self):
        array = np.array([[1.0, np.nan], [3.0, 4.0]])
        fill_value = 666.0
        result = array_nans_to_masked(array, fill_value=fill_value,
                                      filled=True)
        self.assertFalse(isinstance(result, ma.MaskedArray))
        self.assertIs(result, array)
        expected = np.array([[1.0, fill_value], [3.0, 4.0]])
        self.assertArrayEqual(result, expected)

    def test_filled_with_no_fill_value(self):
        array = np.array([[1.0, np.nan], [3.0, 4.0]])
        emsg = 'Invalid fill value'
        with self.assertRaisesRegexp(ValueError, emsg):
            array_nans_to_masked(array, filled=True)

    def test_dtype_cast_float_to_int(self):
        array = np.array([[1.0, np.nan], [3.0, 4.0]])
        dtype = np.int
        result = array_nans_to_masked(array, dtype=dtype)
        self.assertTrue(isinstance(result, ma.MaskedArray))
        expected = ma.masked_array([[1, 2], [3, 4]],
                                   dtype=dtype)
        expected[0, 1] = ma.masked
        self.assertArrayEqual(result, expected)

    def test_filled_dtype_cast_float_to_int(self):
        array = np.array([[1.0, np.nan], [3.0, 4.0]])
        fill_value = 666
        dtype = np.int
        result = array_nans_to_masked(array, fill_value=fill_value,
                                      dtype=dtype, filled=True)
        self.assertFalse(isinstance(result, ma.MaskedArray))
        expected = np.array([[1, fill_value], [3, 4]],
                            dtype=dtype)
        self.assertArrayEqual(result, expected)


if __name__ == '__main__':
    tests.main()
