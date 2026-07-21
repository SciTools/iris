# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.lazy_elementwise`."""

import numpy as np

from iris._lazy_data import as_lazy_data, is_lazy_data, lazy_elementwise
from iris.tests._shared_utils import assert_array_all_close


def _test_elementwise_op(array):
    # Promotes the type of a bool argument, but not a float.
    return array + 1


class Test_lazy_elementwise:
    def test_basic(self):
        concrete_array = np.arange(30).reshape((2, 5, 3))
        lazy_array = as_lazy_data(concrete_array)
        wrapped = lazy_elementwise(lazy_array, _test_elementwise_op)
        assert is_lazy_data(wrapped)
        assert_array_all_close(wrapped.compute(), _test_elementwise_op(concrete_array))

    def test_dtype_same(self):
        concrete_array = np.array([3.0], dtype=np.float16)
        lazy_array = as_lazy_data(concrete_array)
        wrapped = lazy_elementwise(lazy_array, _test_elementwise_op)
        assert is_lazy_data(wrapped)
        assert wrapped.dtype == np.float16
        assert wrapped.compute().dtype == np.float16

    def test_dtype_change(self):
        concrete_array = np.array([True, False])
        lazy_array = as_lazy_data(concrete_array)
        wrapped = lazy_elementwise(lazy_array, _test_elementwise_op)
        assert is_lazy_data(wrapped)
        assert wrapped.dtype == np.int_
        assert wrapped.compute().dtype == wrapped.dtype

    def test_scalar_returning_op_on_0d(self):
        # An op that returns a Python scalar for a 0-d block (e.g.
        # cf_units.Unit.convert on a scalar array) must still yield an array,
        # so that the result can be stored by Dask (#6965).
        def scalar_op(array):
            # Mimics returning a Python float for a scalar input.
            return float(array) if array.ndim == 0 else array + 1.0

        lazy_array = as_lazy_data(np.array(3.0))
        assert lazy_array.ndim == 0
        wrapped = lazy_elementwise(lazy_array, scalar_op)
        assert is_lazy_data(wrapped)
        result = wrapped.compute()
        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        assert result[()] == 3.0
