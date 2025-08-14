# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :mod:`iris.tests._shared_utils`."""

from abc import ABCMeta, abstractmethod

import numpy as np
import pytest

from iris.tests import _shared_utils


class _MaskedArrayEquality(metaclass=ABCMeta):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.arr1 = np.ma.array([1, 2, 3, 4], mask=[False, True, True, False])
        self.arr2 = np.ma.array([1, 3, 2, 4], mask=[False, True, True, False])

    @property
    @abstractmethod
    def _func(self):
        pass

    def test_strict_comparison(self):
        # Comparing both mask and data array completely.
        with pytest.raises(AssertionError):
            self._func(self.arr1, self.arr2, strict=True)

    def test_non_strict_comparison(self):
        # Checking masked array equality and all unmasked array data values.
        self._func(self.arr1, self.arr2, strict=False)

    def test_default_strict_arg_comparison(self):
        self._func(self.arr1, self.arr2)

    def test_nomask(self):
        # Test that an assertion is raised when comparing a masked array
        # containing masked and unmasked values with a masked array with
        # 'nomask'.
        arr1 = np.ma.array([1, 2, 3, 4])
        with pytest.raises(AssertionError):
            self._func(arr1, self.arr2, strict=False)

    def test_nomask_unmasked(self):
        # Ensure that a masked array with 'nomask' can compare with an entirely
        # unmasked array.
        arr1 = np.ma.array([1, 2, 3, 4])
        arr2 = np.ma.array([1, 2, 3, 4], mask=False)
        self._func(arr1, arr2, strict=False)

    def test_different_mask_strict(self):
        # Differing masks, equal data
        arr2 = self.arr1.copy()
        arr2[0] = np.ma.masked
        with pytest.raises(AssertionError):
            self._func(self.arr1, arr2, strict=True)

    def test_different_mask_nonstrict(self):
        # Differing masks, equal data
        arr2 = self.arr1.copy()
        arr2[0] = np.ma.masked
        with pytest.raises(AssertionError):
            self._func(self.arr1, arr2, strict=False)


class Test_assert_masked_array_equal(_MaskedArrayEquality):
    @property
    def _func(self):
        return _shared_utils.assert_masked_array_equal


class Test_assert_masked_array_equal__Nonmaasked:
    def test_nonmasked_same(self):
        # Masked test can be used on non-masked arrays.
        arr1 = np.array([1, 2])
        _shared_utils.assert_masked_array_equal(arr1, arr1)

    def test_masked_nonmasked_same(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=None.
        arr1 = np.ma.masked_array([1, 2])
        arr2 = np.array([1, 2])
        _shared_utils.assert_masked_array_equal(arr1, arr2)

    def test_masked_nonmasked_different(self):
        arr1 = np.ma.masked_array([1, 2])
        arr2 = np.array([1, 3])
        with pytest.raises(AssertionError, match="Arrays are not equal"):
            _shared_utils.assert_masked_array_equal(arr1, arr2)

    def test_nonmasked_masked_same(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=None.
        arr1 = np.array([1, 2])
        arr2 = np.ma.masked_array([1, 2])
        _shared_utils.assert_masked_array_equal(arr1, arr2)

    def test_masked_nonmasked_same_falsemask(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=False.
        arr1 = np.ma.masked_array([1, 2], mask=False)
        arr2 = np.array([1, 2])
        _shared_utils.assert_masked_array_equal(arr1, arr2)

    def test_masked_nonmasked_same_emptymask(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=zeros.
        arr1 = np.ma.masked_array([1, 2], mask=[False, False])
        arr2 = np.array([1, 2])
        _shared_utils.assert_masked_array_equal(arr1, arr2)


class Test_assert_masked_array_almost_equal(_MaskedArrayEquality):
    @property
    def _func(self):
        return _shared_utils.assert_masked_array_almost_equal

    def test_decimal(self):
        emsg = r"Arrays are not almost equal to 3 decimals"
        arr1, arr2 = np.ma.array([100.0]), np.ma.array([100.003])
        _shared_utils.assert_masked_array_almost_equal(arr1, arr2, decimal=2)
        with pytest.raises(AssertionError, match=emsg):
            _shared_utils.assert_masked_array_almost_equal(arr1, arr2, decimal=3)
