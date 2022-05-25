# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util._mask_array"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import dask.array as da
import numpy as np
import numpy.ma as ma
import pytest

import iris._lazy_data
from iris.util import _mask_array

# Set up some arrays to use through the tests.
array_1d = np.arange(4)
masked_arr_1d = ma.array(np.arange(4), mask=[1, 0, 0, 1])

# Any masked points on the mask itself should be ignored.  So result with mask_1d
# and masked_mask_1d should be the same.
mask_1d = np.array([0, 1, 0, 1])
masked_mask_1d = ma.array([0, 1, 1, 1], mask=[0, 0, 1, 0])

# Expected output depends whether input array is masked or not.
expected1 = ma.array(array_1d, mask=mask_1d)
expected2 = ma.array(array_1d, mask=[1, 1, 0, 1])
array_choices = [(array_1d, expected1), (masked_arr_1d, expected2)]


@pytest.mark.parametrize(
    "mask", [mask_1d, masked_mask_1d], ids=["plain-mask", "masked-mask"]
)
@pytest.mark.parametrize("lazy_mask", [False, True], ids=["real", "lazy"])
@pytest.mark.parametrize(
    "array, expected", array_choices, ids=["plain-array", "masked-array"]
)
@pytest.mark.parametrize("lazy_array", [False, True], ids=["real", "lazy"])
def test_1d_not_in_place(array, mask, expected, lazy_array, lazy_mask):
    """
    Basic test for expected behaviour when working not in place with various
    array types for input.

    """
    if lazy_array:
        array = iris._lazy_data.as_lazy_data(array)

    if lazy_mask:
        mask = iris._lazy_data.as_lazy_data(mask)

    result = _mask_array(array, mask)

    if lazy_array or lazy_mask:
        assert iris._lazy_data.is_lazy_data(result)
        result = iris._lazy_data.as_concrete_data(result)

    np.testing.assert_array_equal(expected.mask, result.mask)
    np.testing.assert_array_equal(expected.compressed(), result.compressed())


class Test1DInPlace(tests.IrisTest):
    """Simple 1D in-place cases to check how various types of array interact."""

    def test_plain_array(self):
        arr = np.arange(4)
        mask = None
        with self.assertRaisesRegex(
            TypeError, "Cannot apply a mask in-place to a plain numpy array."
        ):
            _mask_array(arr, mask, in_place=True)

    def test_masked_array_plain_mask(self):
        arr = ma.array(range(4))
        mask = np.array([0, 1, 0, 1])
        expected = ma.array(arr.data, mask=mask)
        result = _mask_array(arr, mask, in_place=True)
        self.assertMaskedArrayEqual(arr, expected)
        # Resolve uses returned value regardless of whether we're working
        # in_place.
        self.assertMaskedArrayEqual(result, expected)

    def test_masked_array_lazy_mask(self):
        arr = ma.array(np.arange(4))
        mask = da.from_array([0, 1, 0, 1])
        with self.assertRaisesRegex(
            TypeError, "Cannot apply lazy mask in-place to a non-lazy array."
        ):
            _mask_array(arr, mask, in_place=True)

    def test_lazy_array(self):
        arr = da.from_array(np.arange(4))
        mask = np.array([0, 1, 0, 1])
        expected_computed = ma.array(range(4), mask=[0, 1, 0, 1])
        # in_place is ignored for lazy array as this is handled by
        # _math_op_common.
        result = _mask_array(arr, mask, in_place=True)
        self.assertTrue(iris._lazy_data.is_lazy_data(result))
        self.assertMaskedArrayEqual(result.compute(), expected_computed)


class TestBroadcast(tests.IrisTest):
    def setUp(self):
        self.arr2by3 = np.arange(6).reshape(2, 3)

    def test_trailing_mask(self):
        arr = self.arr2by3
        mask = np.array([0, 1, 0])
        expected = ma.array(arr, mask=[[0, 1, 0], [0, 1, 0]])
        result = _mask_array(arr, mask, in_place=False)
        self.assertMaskedArrayEqual(result, expected)

    def test_leading_mask(self):
        arr = ma.masked_array(self.arr2by3, mask=[[0, 0, 0], [0, 0, 1]])
        mask = np.array([1, 0]).reshape(2, 1)
        expected = ma.array(arr.data, mask=[[1, 1, 1], [0, 0, 1]])
        result = _mask_array(arr, mask, in_place=False)
        self.assertMaskedArrayEqual(result, expected)

    def test_lazy_trailing_mask(self):
        arr = da.ma.masked_array(self.arr2by3, mask=[[0, 1, 1], [0, 0, 0]])
        mask = np.array([0, 1, 0])
        expected_computed = ma.array(self.arr2by3, mask=[[0, 1, 1], [0, 1, 0]])
        result = _mask_array(arr, mask, in_place=False)
        self.assertMaskedArrayEqual(result.compute(), expected_computed)

    def test_lazy_leading_mask(self):
        arr = da.from_array(self.arr2by3)
        mask = da.from_array([0, 1]).reshape(2, 1)
        expected_computed = ma.array(self.arr2by3, mask=[[0, 0, 0], [1, 1, 1]])
        result = _mask_array(arr, mask, in_place=False)
        self.assertMaskedArrayEqual(result.compute(), expected_computed)


if __name__ == "__main__":
    tests.main()
