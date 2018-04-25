# (C) British Crown Copyright 2018, Met Office
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
"""Test function :func:`iris._lazy data.lazy_elementwise`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris._lazy_data import as_lazy_data, is_lazy_data

from iris._lazy_data import lazy_elementwise


def _test_elementwise_op(array):
    # Promotes the type of a bool argument, but not a float.
    return array + 1


class Test_lazy_elementwise(tests.IrisTest):
    def test_basic(self):
        concrete_array = np.arange(30).reshape((2, 5, 3))
        lazy_array = as_lazy_data(concrete_array)
        wrapped = lazy_elementwise(lazy_array, _test_elementwise_op)
        self.assertTrue(is_lazy_data(wrapped))
        self.assertArrayAllClose(wrapped.compute(),
                                 _test_elementwise_op(concrete_array))

    def test_dtype_same(self):
        concrete_array = np.array([3.], dtype=np.float16)
        lazy_array = as_lazy_data(concrete_array)
        wrapped = lazy_elementwise(lazy_array, _test_elementwise_op)
        self.assertTrue(is_lazy_data(wrapped))
        self.assertEqual(wrapped.dtype, np.float16)
        self.assertEqual(wrapped.compute().dtype, np.float16)

    def test_dtype_change(self):
        concrete_array = np.array([True, False])
        lazy_array = as_lazy_data(concrete_array)
        wrapped = lazy_elementwise(lazy_array, _test_elementwise_op)
        self.assertTrue(is_lazy_data(wrapped))
        self.assertEqual(wrapped.dtype, np.int)
        self.assertEqual(wrapped.compute().dtype, wrapped.dtype)


if __name__ == '__main__':
    tests.main()
