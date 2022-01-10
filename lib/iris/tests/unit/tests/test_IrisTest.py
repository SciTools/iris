# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris.tests.IrisTest` class."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from abc import ABCMeta, abstractmethod

import numpy as np


class _MaskedArrayEquality(metaclass=ABCMeta):
    def setUp(self):
        self.arr1 = np.ma.array([1, 2, 3, 4], mask=[False, True, True, False])
        self.arr2 = np.ma.array([1, 3, 2, 4], mask=[False, True, True, False])

    @property
    @abstractmethod
    def _func(self):
        pass

    def test_strict_comparison(self):
        # Comparing both mask and data array completely.
        with self.assertRaises(AssertionError):
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
        with self.assertRaises(AssertionError):
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
        with self.assertRaises(AssertionError):
            self._func(self.arr1, arr2, strict=True)

    def test_different_mask_nonstrict(self):
        # Differing masks, equal data
        arr2 = self.arr1.copy()
        arr2[0] = np.ma.masked
        with self.assertRaises(AssertionError):
            self._func(self.arr1, arr2, strict=False)


@tests.iristest_timing_decorator
class Test_assertMaskedArrayEqual(_MaskedArrayEquality, tests.IrisTest_nometa):
    @property
    def _func(self):
        return self.assertMaskedArrayEqual


class Test_assertMaskedArrayEqual__Nonmaasked(tests.IrisTest):
    def test_nonmasked_same(self):
        # Masked test can be used on non-masked arrays.
        arr1 = np.array([1, 2])
        self.assertMaskedArrayEqual(arr1, arr1)

    def test_masked_nonmasked_same(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=None.
        arr1 = np.ma.masked_array([1, 2])
        arr2 = np.array([1, 2])
        self.assertMaskedArrayEqual(arr1, arr2)

    def test_masked_nonmasked_different(self):
        arr1 = np.ma.masked_array([1, 2])
        arr2 = np.array([1, 3])
        with self.assertRaisesRegex(AssertionError, "Arrays are not equal"):
            self.assertMaskedArrayEqual(arr1, arr2)

    def test_nonmasked_masked_same(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=None.
        arr1 = np.array([1, 2])
        arr2 = np.ma.masked_array([1, 2])
        self.assertMaskedArrayEqual(arr1, arr2)

    def test_masked_nonmasked_same_falsemask(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=False.
        arr1 = np.ma.masked_array([1, 2], mask=False)
        arr2 = np.array([1, 2])
        self.assertMaskedArrayEqual(arr1, arr2)

    def test_masked_nonmasked_same_emptymask(self):
        # Masked test can be used between masked + non-masked arrays, and will
        # consider them equal, when mask=zeros.
        arr1 = np.ma.masked_array([1, 2], mask=[False, False])
        arr2 = np.array([1, 2])
        self.assertMaskedArrayEqual(arr1, arr2)


@tests.iristest_timing_decorator
class Test_assertMaskedArrayAlmostEqual(
    _MaskedArrayEquality, tests.IrisTest_nometa
):
    @property
    def _func(self):
        return self.assertMaskedArrayAlmostEqual

    def test_decimal(self):
        arr1, arr2 = np.ma.array([100.0]), np.ma.array([100.003])
        self.assertMaskedArrayAlmostEqual(arr1, arr2, decimal=2)
        with self.assertRaises(AssertionError):
            self.assertMaskedArrayAlmostEqual(arr1, arr2, decimal=3)


if __name__ == "__main__":
    tests.main()
