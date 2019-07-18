# (C) British Crown Copyright 2017 - 2019, Met Office
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
"""Test function :func:`iris._lazy data.as_concrete_data`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
from iris.tests import mock


class MyProxy(object):
    def __init__(self, a):
        self.shape = a.shape
        self.dtype = a.dtype
        self.ndim = a.ndim
        self.a = a

    def __getitem__(self, keys):
        return self.a[keys]


class Test_as_concrete_data(tests.IrisTest):
    def test_concrete_input_data(self):
        data = np.arange(24).reshape((4, 6))
        result = as_concrete_data(data)
        self.assertIs(data, result)
        self.assertFalse(is_lazy_data(result))

    def test_concrete_masked_input_data(self):
        data = ma.masked_array([10, 12, 8, 2], mask=[True, True, False, True])
        result = as_concrete_data(data)
        self.assertIs(data, result)
        self.assertFalse(is_lazy_data(result))

    def test_lazy_data(self):
        data = np.arange(24).reshape((2, 12))
        lazy_array = as_lazy_data(data)
        self.assertTrue(is_lazy_data(lazy_array))
        result = as_concrete_data(lazy_array)
        self.assertFalse(is_lazy_data(result))
        self.assertArrayEqual(result, data)

    def test_lazy_mask_data(self):
        data = np.arange(24).reshape((2, 12))
        fill_value = 1234
        mask_data = ma.masked_array(data, fill_value=fill_value)
        lazy_array = as_lazy_data(mask_data)
        self.assertTrue(is_lazy_data(lazy_array))
        result = as_concrete_data(lazy_array)
        self.assertFalse(is_lazy_data(result))
        self.assertMaskedArrayEqual(result, mask_data)
        self.assertEqual(result.fill_value, fill_value)

    def test_lazy_scalar_proxy(self):
        a = np.array(5)
        proxy = MyProxy(a)
        lazy_array = as_lazy_data(proxy)
        self.assertTrue(is_lazy_data(lazy_array))
        result = as_concrete_data(lazy_array)
        self.assertFalse(is_lazy_data(result))
        self.assertEqual(result, a)

    def test_lazy_scalar_proxy_masked(self):
        a = np.ma.masked_array(5, True)
        proxy = MyProxy(a)
        lazy_array = as_lazy_data(proxy)
        self.assertTrue(is_lazy_data(lazy_array))
        result = as_concrete_data(lazy_array)
        self.assertFalse(is_lazy_data(result))
        self.assertMaskedArrayEqual(result, a)


if __name__ == '__main__':
    tests.main()
