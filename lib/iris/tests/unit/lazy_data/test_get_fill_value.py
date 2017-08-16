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
"""Test the function :func:`iris._lazy data.get_fill_value`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_lazy_data, get_fill_value


class Test_get_fill_value(tests.IrisTest):
    def setUp(self):
        # Array shape and fill-value.
        spec = [((2, 3, 4, 5), -1),  # 4d array
                ((2, 3, 4), -2),     # 3d array
                ((2, 3), -3),        # 2d array
                ((2,), -4),          # 1d array
                ((), -5)]            # 0d array
        self.arrays = [np.empty(shape) for (shape, _) in spec]
        self.masked = [ma.empty(shape, fill_value=fv) for (shape, fv) in spec]
        self.lazy_arrays = [as_lazy_data(array) for array in self.arrays]
        self.lazy_masked = [as_lazy_data(array) for array in self.masked]
        # Add the masked constant case.
        mc = ma.array([0], mask=True)[0]
        self.masked.append(mc)
        self.lazy_masked.append(as_lazy_data(mc))
        # Collect the expected fill-values.
        self.expected_fill_values = [fv for (_, fv) in spec]
        mc_fill_value = ma.masked_array(0, dtype=mc.dtype).fill_value
        self.expected_fill_values.append(mc_fill_value)

    def test_arrays(self):
        for array in self.arrays:
            self.assertIsNone(get_fill_value(array))

    def test_masked(self):
        for array, expected in zip(self.masked, self.expected_fill_values):
            result = get_fill_value(array)
            self.assertEqual(result, expected)

    def test_lazy_arrays(self):
        for array in self.lazy_arrays:
            self.assertIsNone(get_fill_value(array))

    def test_lazy_masked(self):
        for array, expected in zip(self.lazy_masked,
                                   self.expected_fill_values):
            result = get_fill_value(array)
            self.assertEqual(result, expected)


if __name__ == '__main__':
    tests.main()
