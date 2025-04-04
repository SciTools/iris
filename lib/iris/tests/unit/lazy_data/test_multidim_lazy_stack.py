# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.multidim_lazy_stack`."""

import dask.array as da
import numpy as np

from iris._lazy_data import as_concrete_data, as_lazy_data, multidim_lazy_stack
from iris.tests._shared_utils import assert_array_all_close


class Test_multidim_lazy_stack:
    def _check(self, stack_shape):
        vals = np.arange(np.prod(stack_shape)).reshape(stack_shape)
        stack = np.empty(stack_shape, "object")
        # Define the shape of each element in the stack.
        stack_element_shape = (4, 5)
        expected = np.empty(stack_shape + stack_element_shape, dtype=int)
        for index, val in np.ndenumerate(vals):
            stack[index] = as_lazy_data(val * np.ones(stack_element_shape))

            expected[index] = val
        result = multidim_lazy_stack(stack)
        assert result.shape == stack_shape + stack_element_shape
        assert isinstance(result, da.core.Array)
        result = as_concrete_data(result)
        assert_array_all_close(result, expected)

    def test_0d_lazy_stack(self):
        shape = ()
        self._check(shape)

    def test_1d_lazy_stack(self):
        shape = (2,)
        self._check(shape)

    def test_2d_lazy_stack(self):
        shape = (3, 2)
        self._check(shape)
