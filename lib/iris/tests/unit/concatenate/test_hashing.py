# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test array hashing in :mod:`iris._concatenate`."""

import dask.array as da
import numpy as np
import pytest

from iris import _concatenate
from iris.tests.unit.util.test_array_equal import TEST_CASES
from iris.util import array_equal


@pytest.mark.parametrize(
    "a,b,eq",
    [
        (np.arange(2), da.arange(2), True),
        (np.arange(2), np.arange(2).reshape((1, 2)), False),
        (da.arange(2), da.arange(2).reshape((1, 2)), False),
        (np.array([1], dtype=np.float32), np.array([1], dtype=bool), True),
        (np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), True),
        (np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 1]), True),
        (np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 0]), False),
        (np.ma.array([1, 2], mask=[1, 1]), np.ma.array([1, 2], mask=[1, 1]), True),
        (np.ma.array([1, 2], mask=[0, 0]), np.ma.array([1, 2], mask=[0, 0]), True),
        (da.arange(6).reshape((2, 3)), da.arange(6, chunks=1).reshape((2, 3)), True),
        (da.arange(20, chunks=1), da.arange(20, chunks=2), True),
        (
            da.ma.masked_array([1, 2], mask=[0, 1]),
            da.ma.masked_array([1, 2], mask=[0, 1]),
            True,
        ),
        (
            da.ma.masked_array([1, 2], mask=[0, 1]),
            da.ma.masked_array([1, 3], mask=[0, 1]),
            True,
        ),
        (
            np.arange(2),
            da.ma.masked_array(np.arange(2), mask=[0, 0]),
            True,
        ),
        (
            np.arange(2),
            da.ma.masked_array(np.arange(2), mask=[0, 1]),
            False,
        ),
        (
            da.ma.masked_array(np.arange(10), mask=np.zeros(10)),
            da.ma.masked_array(np.arange(10), mask=np.ma.nomask),
            True,
        ),
        (
            np.ma.array([1, 2], mask=[0, 1]),
            np.ma.array([1, 3], mask=[0, 1], fill_value=10),
            True,
        ),
        (
            np.ma.masked_array([1], mask=[True]),
            np.array([np.ma.default_fill_value(np.dtype("int64"))]),
            False,
        ),
        (np.array(["a", "b"]), np.array(["a", "b"]), True),
        (np.array(["a"]), np.array(["b"]), False),
        (da.asarray(["a", "b"], chunks=1), da.asarray(["a", "b"], chunks=1), True),
        (da.array(["a"]), da.array(["b"]), False),
        (np.array(["a"]), da.array(["a"]), True),
        (np.array(["a"]), np.array([1]), False),
        (da.asarray([1, 1], chunks=1), da.asarray(["a", "b"], chunks=1), False),
        (np.array(["a"]), np.array(["a"]).view(dtype=np.int32), False),
    ],
)
def test_compute_hashes(a, b, eq):
    hashes = _concatenate._compute_hashes({"a": a, "b": b})
    assert eq == (hashes["a"] == hashes["b"])


@pytest.mark.parametrize(
    "a,b",
    [
        (a, b)
        for (a, b, withnans, eq) in TEST_CASES
        if isinstance(a, np.ndarray | da.Array) and isinstance(b, np.ndarray | da.Array)
    ],
)
def test_compute_hashes_vs_array_equal(a, b):
    """Test that hashing give the same answer as `array_equal(withnans=True)`."""
    hashes = _concatenate._compute_hashes({"a": a, "b": b})
    assert array_equal(a, b, withnans=True) == (hashes["a"] == hashes["b"])


def test_arrayhash_equal_incompatible_chunks_raises():
    hash1 = _concatenate._ArrayHash(1, chunks=((1, 1),))
    hash2 = _concatenate._ArrayHash(1, chunks=((2,),))
    msg = r"Unable to compare arrays with different chunks.*"
    with pytest.raises(ValueError, match=msg):
        hash1 == hash2


def test_arrayhash_equal_incompatible_type_raises():
    hash = _concatenate._ArrayHash(1, chunks=(1, 1))
    msg = r"Unable to compare .*"
    with pytest.raises(TypeError, match=msg):
        hash == object()
