# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.array_equal`."""

import dask.array as da
import numpy as np
import numpy.ma as ma
import pytest

from iris.util import array_equal

ARRAY1 = np.array(np.arange(24).reshape(2, 3, 4))
ARRAY1[0, 1, 2] = 100

ARRAY2 = np.array([1.0, np.nan, 2.0, np.nan, 3.0])

TEST_CASES = [
    # test empty
    (np.array([]), np.array([]), False, True),
    (np.array([]), np.array([], dtype=np.float64), True, True),
    # test 0d
    (np.array(23), np.array(23), False, True),
    (np.array(23), np.array(7), False, False),
    # test 0d and scalar
    (np.array(23), 23, False, True),
    (np.array(23), 45, False, False),
    # test 1d and sequences
    (np.array([1, 2, 3]), [1, 2, 3], False, True),
    (np.array([1, 2, 3]), [1, 2], False, False),
    (np.array([1, 45, 3]), [1, 2, 3], False, False),
    (np.array([1, 2, 3]), (1, 2, 3), False, True),
    (np.array([1, 2, 3]), (1, 2), False, False),
    (np.array([1, 45, 3]), (1, 2, 3), False, False),
    # test 3d
    (
        np.array(np.arange(24).reshape(2, 3, 4)),
        np.array(np.arange(24).reshape(2, 3, 4)),
        False,
        True,
    ),
    (
        np.array(np.arange(24).reshape(2, 3, 4)),
        ARRAY1,
        False,
        False,
    ),
    # test masked is not ignored
    (
        ma.masked_array([1, 2, 3], mask=[1, 0, 1]),
        ma.masked_array([2, 2, 2], mask=[1, 0, 1]),
        False,
        True,
    ),
    # test masked is different
    (
        ma.masked_array([1, 2, 3], mask=[1, 0, 1]),
        ma.masked_array([1, 2, 3], mask=[0, 0, 1]),
        False,
        False,
    ),
    # test masked isn't unmasked
    (
        np.array([1, 2, 2]),
        ma.masked_array([1, 2, 2], mask=[0, 0, 1]),
        False,
        False,
    ),
    (
        ma.masked_array([1, 2, 2], mask=[0, 0, 1]),
        ma.masked_array([1, 2, 2]),
        False,
        False,
    ),
    (
        np.array([1, 2]),
        ma.masked_array([1, 3], mask=[0, 1]),
        False,
        False,
    ),
    # test masked/unmasked_equivalence
    (
        np.array([1, 2, 2]),
        ma.masked_array([1, 2, 2]),
        False,
        True,
    ),
    (
        np.array([1, 2, 2]),
        ma.masked_array([1, 2, 2], mask=[0, 0, 0]),
        False,
        True,
    ),
    # test fully masked arrays
    (
        ma.masked_array(np.arange(24).reshape(2, 3, 4), mask=True),
        ma.masked_array(np.arange(24).reshape(2, 3, 4), mask=True),
        False,
        True,
    ),
    # test fully masked 0d arrays
    (
        ma.masked_array(3, mask=True),
        ma.masked_array(3, mask=True),
        False,
        True,
    ),
    # test fully masked string arrays
    (
        ma.masked_array(["a", "b", "c"], mask=True),
        ma.masked_array(["a", "b", "c"], mask=[1, 1, 1]),
        False,
        True,
    ),
    # test partially masked string arrays
    (
        ma.masked_array(["a", "b", "c"], mask=[1, 0, 1]),
        ma.masked_array(["a", "b", "c"], mask=[1, 0, 1]),
        False,
        True,
    ),
    # test string arrays equal
    (
        np.array(["abc", "def", "efg"]),
        np.array(["abc", "def", "efg"]),
        False,
        True,
    ),
    # test string arrays different contents
    (
        np.array(["abc", "def", "efg"]),
        np.array(["abc", "de", "efg"]),
        False,
        False,
    ),
    # test string arrays subset
    (
        np.array(["abc", "def", "efg"]),
        np.array(["abc", "def"]),
        False,
        False,
    ),
    (
        np.array(["abc", "def"]),
        np.array(["abc", "def", "efg"]),
        False,
        False,
    ),
    # test string arrays unequal dimensionality
    (np.array("abc"), np.array(["abc"]), False, False),
    (np.array(["abc"]), np.array("abc"), False, False),
    (np.array("abc"), np.array([["abc"]]), False, False),
    (np.array(["abc"]), np.array([["abc"]]), False, False),
    # test string arrays 0d and scalar
    (np.array("foobar"), "foobar", False, True),
    (np.array("foobar"), "foo", False, False),
    (np.array("foobar"), "foobar.", False, False),
    # test nan equality nan ne nan
    (ARRAY2, ARRAY2, False, False),
    (ARRAY2, ARRAY2.copy(), False, False),
    # test nan equality nan naneq nan
    (ARRAY2, ARRAY2, True, True),
    (ARRAY2, ARRAY2.copy(), True, True),
    # test nan equality nan nanne a
    (
        np.array([1.0, np.nan, 2.0, np.nan, 3.0]),
        np.array([1.0, np.nan, 2.0, 0.0, 3.0]),
        True,
        False,
    ),
    # test nan equality a nanne b
    (
        np.array([1.0, np.nan, 2.0, np.nan, 3.0]),
        np.array([1.0, np.nan, 2.0, np.nan, 4.0]),
        True,
        False,
    ),
]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("array_a,array_b,withnans,eq", TEST_CASES)
def test_array_equal(array_a, array_b, withnans, eq, lazy):
    if lazy:
        identical = array_a is array_b
        if isinstance(array_a, np.ndarray):
            array_a = da.asarray(array_a, chunks=2)
        if isinstance(array_b, np.ndarray):
            array_b = da.asarray(array_b, chunks=1)
        if identical:
            array_b = array_a
    assert eq == array_equal(array_a, array_b, withnans=withnans)
