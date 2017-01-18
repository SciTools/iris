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
Test lazy data utility functions.

Note: really belongs in "tests/unit/lazy_data".

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests


import numpy as np
import dask.array as da


from iris._lazy_data import is_lazy_data, as_lazy_data, as_concrete_data


class MixinLazyTests(object):
    def setUp(self):
        # Create test real and dask arrays.
        self.real_array = np.arange(24).reshape((2, 3, 4))
        self.lazy_values = np.arange(30).reshape((2, 5, 3))
        self.lazy_array = da.from_array(self.lazy_values, 1e6)


class Test_is_lazy_data(MixinLazyTests, tests.IrisTest):
    def test_lazy(self):
        self.assertTrue(is_lazy_data(self.lazy_array))

    def test_real(self):
        self.assertFalse(is_lazy_data(self.real_array))


class Test_as_lazy_data(MixinLazyTests, tests.IrisTest):
    def test_lazy(self):
        result = as_lazy_data(self.lazy_array)
        self.assertTrue(is_lazy_data(result))
        self.assertIs(result, self.lazy_array)

    def test_real(self):
        result = as_lazy_data(self.real_array)
        self.assertTrue(is_lazy_data(result))
        self.assertArrayAllClose(as_concrete_data(result), self.real_array)


class Test_as_concrete_data(MixinLazyTests, tests.IrisTest):
    def test_lazy(self):
        result = as_concrete_data(self.lazy_array)
        self.assertFalse(is_lazy_data(result))
        self.assertArrayAllClose(result, self.lazy_values)

    def test_real(self):
        result = as_concrete_data(self.real_array)
        self.assertFalse(is_lazy_data(result))
        self.assertIs(result, self.real_array)


if __name__ == '__main__':
    tests.main()
