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
"""Test function :func:`iris._lazy data.array_masked_to_nans`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests


import numpy as np
import numpy.ma as ma

from iris._lazy_data import array_masked_to_nans


class Test(tests.IrisTest):
    def _common_checks(self, result):
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(ma.isMaskedArray(result))

    def test_masked(self):
        masked_array = ma.masked_array([[1.0, 2.0], [3.0, 4.0]],
                                       mask=[[0, 1], [0, 0]])

        result = array_masked_to_nans(masked_array)

        self._common_checks(result)
        self.assertArrayAllClose(np.isnan(result),
                                 [[False, True], [False, False]])
        result[0, 1] = 777.7
        self.assertArrayAllClose(result, [[1.0, 777.7], [3.0, 4.0]])

    def test_unmasked(self):
        unmasked_array = np.array([1.0, 2.0])
        result = array_masked_to_nans(unmasked_array)
        # Non-masked array is returned as-is, without copying.
        self.assertIs(result, unmasked_array)

    def test_empty_mask(self):
        masked_array = ma.masked_array([1.0, 2.0], mask=[0, 0])

        result = array_masked_to_nans(masked_array)

        self._common_checks(result)
        self.assertArrayAllClose(result, masked_array.data)

    def test_no_mask(self):
        masked_array = ma.masked_array([1.0, 2.0], mask=ma.nomask)

        result = array_masked_to_nans(masked_array)

        self._common_checks(result)
        self.assertArrayAllClose(result, masked_array.data)

    def test_masked__integers(self):
        masked_array = ma.masked_array([[1, 2], [3, 4]],
                                       mask=[[0, 1], [0, 0]])

        result = array_masked_to_nans(masked_array)

        self._common_checks(result)
        self.assertEqual(result.dtype, np.dtype('f8'))
        self.assertArrayAllClose(np.isnan(result),
                                 [[False, True], [False, False]])
        result[0, 1] = 777.7
        self.assertArrayAllClose(result, [[1.0, 777.7], [3.0, 4.0]])

    def test_unmasked__integers(self):
        unmasked_array = np.array([1, 2])
        result = array_masked_to_nans(unmasked_array)
        # Non-masked array is returned as-is, without copying.
        self.assertIs(result, unmasked_array)

    def test_no_mask__integers(self):
        masked_array = ma.masked_array([1, 2], mask=ma.nomask)

        result = array_masked_to_nans(masked_array)

        self._common_checks(result)
        self.assertEqual(result.dtype, np.dtype('f8'))


if __name__ == '__main__':
    tests.main()
