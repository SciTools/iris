# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test array hashing in :mod:`iris._concatenate`."""

import dask.array as da
from dask.base import tokenize
import numpy as np
import pytest

from iris import _concatenate


@pytest.mark.parametrize(
    "a,b,eq",
    [
        (np.arange(2), da.arange(2), True),
        (np.array([1], dtype=np.float32), np.array([1], dtype=bool), True),
        (np.array([1]), np.array([[1]]), False),
        (np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 1]), True),
        (da.arange(2, chunks=1), da.arange(2, chunks=2), True),
    ],
)
def test_compute_hashes(a, b, eq):
    hashes = _concatenate._compute_hashes([a, b])
    assert eq == (hashes[tokenize(a)].value == hashes[tokenize(b)].value)
