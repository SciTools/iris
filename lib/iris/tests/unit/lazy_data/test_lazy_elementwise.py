# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris._lazy data.lazy_elementwise`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris._lazy_data import as_lazy_data, is_lazy_data, lazy_elementwise


def _test_elementwise_op(array):
    # Promotes the type of a bool argument, but not a float.
    return array + 1


class Test_lazy_elementwise(tests.IrisTest):
    def test_basic(self):
        concrete_array = np.arange(30).reshape((2, 5, 3))
        lazy_array = as_lazy_data(concrete_array)
        wrapped = lazy_elementwise(lazy_array, _test_elementwise_op)
        self.assertTrue(is_lazy_data(wrapped))
        self.assertArrayAllClose(
            wrapped.compute(), _test_elementwise_op(concrete_array)
        )

    def test_dtype_same(self):
        concrete_array = np.array([3.0], dtype=np.float16)
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
        self.assertEqual(wrapped.dtype, np.int_)
        self.assertEqual(wrapped.compute().dtype, wrapped.dtype)


if __name__ == "__main__":
    tests.main()
