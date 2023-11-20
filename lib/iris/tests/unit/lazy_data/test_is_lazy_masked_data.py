# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.is_lazy_masked_data`."""

import dask.array as da
import numpy as np
import pytest

from iris._lazy_data import is_lazy_masked_data

real_arrays = [
    np.arange(3),
    np.ma.array(range(3)),
    np.ma.array(range(3), mask=[0, 1, 1]),
]
lazy_arrays = [da.from_array(arr) for arr in real_arrays]


@pytest.mark.parametrize(
    "arr, expected", zip(real_arrays + lazy_arrays, [False] * 4 + [True] * 2)
)
def test_is_lazy_masked_data(arr, expected):
    result = is_lazy_masked_data(arr)
    assert result is expected
