# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.non_lazy`."""

import numpy as np

from iris._lazy_data import as_lazy_data, is_lazy_data, non_lazy
from iris.tests._shared_utils import assert_array_equal


class Test_non_lazy:
    def setup_method(self):
        self.array = np.arange(8).reshape(2, 4)
        self.lazy_array = as_lazy_data(self.array)
        self.func = non_lazy(lambda array: array.sum(axis=0))
        self.func_result = [4, 6, 8, 10]

    def test_lazy_input(self):
        result = self.func(self.lazy_array)
        assert not is_lazy_data(result)
        assert_array_equal(result, self.func_result)

    def test_non_lazy_input(self):
        # Check that a non-lazy input doesn't trip up the functionality.
        result = self.func(self.array)
        assert not is_lazy_data(result)
        assert_array_equal(result, self.func_result)
