# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util._mask_array"""

import dask.array as da
import numpy as np
import numpy.ma as ma
import pytest

import iris._lazy_data
from iris.tests import assert_masked_array_equal
from iris.util import _mask_array

# Set up some arrays to use through the tests.
array_1d = np.arange(4)
masked_arr_1d = ma.array(np.arange(4), mask=[1, 0, 0, 1])
array_2by3 = np.arange(6).reshape(2, 3)

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
    assert result is not array

    if lazy_array or lazy_mask:
        assert iris._lazy_data.is_lazy_data(result)
        result = iris._lazy_data.as_concrete_data(result)

    assert_masked_array_equal(expected, result)


# 1D in place tests.


def test_plain_array_in_place():
    """
    Test we get an informative error when trying to add a mask to a plain numpy
    array.

    """
    arr = array_1d
    mask = None
    with pytest.raises(
        TypeError, match="Cannot apply a mask in-place to a plain numpy array."
    ):
        _mask_array(arr, mask, in_place=True)


def test_masked_array_lazy_mask_in_place():
    """
    Test we get an informative error when trying to apply a lazy mask in-place
    to a non-lazy array.

    """
    arr = masked_arr_1d
    mask = da.from_array([0, 1, 0, 1])
    with pytest.raises(
        TypeError, match="Cannot apply lazy mask in-place to a non-lazy array."
    ):
        _mask_array(arr, mask, in_place=True)


@pytest.mark.parametrize(
    "mask", [mask_1d, masked_mask_1d], ids=["plain-mask", "masked-mask"]
)
def test_real_masked_array_in_place(mask):
    """
    Check expected behaviour for applying masks in-place to a masked array.

    """
    arr = masked_arr_1d.copy()
    result = _mask_array(arr, mask, in_place=True)
    assert_masked_array_equal(arr, expected2)
    # Resolve uses returned value regardless of whether we're working in_place.
    assert result is arr


def test_lazy_array_in_place():
    """
    Test that in place flag is ignored for lazy arrays, and result is the same
    as the not in_place case.

    """
    arr = da.from_array(np.arange(4))
    mask = np.array([0, 1, 0, 1])
    expected_computed = ma.array(range(4), mask=[0, 1, 0, 1])
    # in_place is ignored for lazy array as this is handled by _math_op_common.
    result = _mask_array(arr, mask, in_place=True)
    assert iris._lazy_data.is_lazy_data(result)
    assert_masked_array_equal(result.compute(), expected_computed)
    assert result is not arr


# Broadcasting tests.

IN_PLACE_PARAMETRIZE = pytest.mark.parametrize(
    "in_place", [False, True], ids=["not-in-place", "in-place"]
)


@IN_PLACE_PARAMETRIZE
def test_trailing_mask(in_place):
    array = ma.array(array_2by3.copy())
    mask = np.array([0, 1, 0])
    expected = ma.array(array_2by3, mask=[[0, 1, 0], [0, 1, 0]])
    result = _mask_array(array, mask, in_place=in_place)
    assert_masked_array_equal(result, expected)
    assert result is array if in_place else result is not array


@IN_PLACE_PARAMETRIZE
def test_leading_mask(in_place):
    arr = ma.masked_array(array_2by3.copy(), mask=[[0, 0, 0], [0, 0, 1]])
    mask = np.array([1, 0]).reshape(2, 1)
    expected = ma.array(arr.data, mask=[[1, 1, 1], [0, 0, 1]])
    result = _mask_array(arr, mask, in_place=in_place)
    assert_masked_array_equal(result, expected)
    assert result is arr if in_place else result is not arr


def test_lazy_trailing_mask():
    arr = da.ma.masked_array(array_2by3, mask=[[0, 1, 1], [0, 0, 0]])
    mask = np.array([0, 1, 0])
    expected_computed = ma.array(array_2by3, mask=[[0, 1, 1], [0, 1, 0]])
    result = _mask_array(arr, mask, in_place=False)
    assert iris._lazy_data.is_lazy_data(result)
    assert_masked_array_equal(result.compute(), expected_computed)
    assert result is not arr


def test_lazy_leading_mask():
    arr = da.from_array(array_2by3)
    mask = da.from_array([0, 1]).reshape(2, 1)
    expected_computed = ma.array(array_2by3, mask=[[0, 0, 0], [1, 1, 1]])
    result = _mask_array(arr, mask, in_place=False)
    assert iris._lazy_data.is_lazy_data(result)
    assert_masked_array_equal(result.compute(), expected_computed)
    assert result is not arr


if __name__ == "__main__":
    pytest.main([__file__])
