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
Test :func:`iris._lazy data.convert_nans_array` function.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests


import numpy as np
import numpy.ma as ma

from iris._lazy_data import convert_nans_array


class Test(tests.IrisTest):
    def setUp(self):
        self.array = np.array([[1.0, np.nan],
                               [3.0, 4.0]])

    def test_pass_thru_masked_array_float(self):
        array = ma.arange(10, dtype=np.float)
        result = convert_nans_array(array)
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertIs(result, array)

    def test_pass_thru_masked_array_integer(self):
        array = ma.arange(10)
        result = convert_nans_array(array)
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertIs(result, array)

    def test_no_nans(self):
        array = np.array([[1.0, 2.0],
                          [3.0, 4.0]])
        result = convert_nans_array(array)
        self.assertNotIsInstance(result, ma.MaskedArray)
        self.assertIs(result, array)

    def test_nans_masked(self):
        result = convert_nans_array(self.array, nans=ma.masked)
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertArrayEqual(result.mask, [[False, True],
                                            [False, False]])
        dummy = 666.0
        result[0, 1] = dummy
        self.assertArrayEqual(result.data, [[1.0, dummy],
                                            [3.0, 4.0]])
        self.assertIsNot(result, self.array)
        # Check that fill value is the "standard" one for the type.
        expected = ma.masked_array(self.array[0], dtype=self.array.dtype)
        self.assertEqual(result.fill_value, expected.fill_value)

    def test_nans_filled(self):
        fill_value = 666.0
        result = convert_nans_array(self.array, nans=fill_value)
        self.assertNotIsInstance(result, ma.MaskedArray)
        self.assertIs(result, self.array)
        expected = np.array([[1.0, fill_value],
                             [3.0, 4.0]])
        self.assertArrayEqual(result, expected)

    def test_nans_filled_failure(self):
        fill_value = 1e+20
        dtype = np.dtype('int16')
        emsg = 'Fill value of .* invalid for array result .*'
        with self.assertRaisesRegexp(ValueError, emsg):
            convert_nans_array(self.array, nans=fill_value,
                               result_dtype=dtype)

    def test_nans_none_failure(self):
        emsg = 'Array contains unexpected NaNs'
        with self.assertRaisesRegexp(ValueError, emsg):
            convert_nans_array(self.array)

    def test_result_dtype_cast_float_to_int(self):
        dtype = np.int
        result = convert_nans_array(self.array, nans=ma.masked,
                                    result_dtype=dtype)
        self.assertIsInstance(result, ma.MaskedArray)
        expected = ma.masked_array([[1, 2],
                                    [3, 4]],
                                   dtype=dtype)
        expected[0, 1] = ma.masked
        self.assertArrayEqual(result, expected)
        self.assertIsNot(result, self.array)

    def test_filled_result_dtype_cast_float_to_int(self):
        fill_value = 666
        dtype = np.int
        result = convert_nans_array(self.array, nans=fill_value,
                                    result_dtype=dtype)
        self.assertNotIsInstance(result, ma.MaskedArray)
        expected = np.array([[1, fill_value],
                             [3, 4]],
                            dtype=dtype)
        self.assertArrayEqual(result, expected)


if __name__ == '__main__':
    tests.main()
