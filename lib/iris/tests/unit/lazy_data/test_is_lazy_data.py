# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.is_lazy_data`."""

import dask.array as da
import numpy as np

from iris._lazy_data import is_lazy_data


class Test_is_lazy_data:
    def test_lazy(self):
        values = np.arange(30).reshape((2, 5, 3))
        lazy_array = da.from_array(values, chunks="auto")
        assert is_lazy_data(lazy_array)

    def test_real(self):
        real_array = np.arange(24).reshape((2, 3, 4))
        assert not is_lazy_data(real_array)
