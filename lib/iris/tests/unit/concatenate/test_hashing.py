# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test array hashing in :mod:`iris._concatenate`."""

import dask.array as da
import numpy as np
import pytest

from iris import _concatenate


@pytest.mark.parametrize(
    "a,b,eq",
    [
        (np.arange(2), da.arange(2), True),
        (np.array([1], dtype=np.float32), np.array([1], dtype=bool), True),
        (np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 1]), True),
        (np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 0]), False),
        (da.arange(6).reshape((2, 3)), da.arange(6, chunks=1).reshape((2, 3)), True),
        (da.arange(20, chunks=1), da.arange(20, chunks=2), True),
        (
            da.ma.masked_array([1, 2], mask=[0, 1]),
            da.ma.masked_array([1, 2], mask=[0, 1]),
            True,
        ),
    ],
)
def test_compute_hashes(a, b, eq):
    hashes = _concatenate._compute_hashes([a, b])
    assert eq == (hashes[_concatenate.array_id(a)] == hashes[_concatenate.array_id(b)])


def test_arrayhash_incompatible_chunks_raises():
    hash1 = _concatenate._ArrayHash(1, chunks=(1, 1))
    hash2 = _concatenate._ArrayHash(1, chunks=(2,))
    with pytest.raises(ValueError):
        hash1 == hash2
