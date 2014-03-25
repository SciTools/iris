# (C) British Crown Copyright 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :mod:`iris.tests.IrisTest` class."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from abc import ABCMeta, abstractproperty

import numpy as np


class _MaskedArrayEquality(object):
    __metaclass__ = ABCMeta

    def setUp(self):
        self.arr1 = np.ma.array([1, 2, 3, 4], mask=[False, True, True, False])
        self.arr2 = np.ma.array([1, 3, 2, 4], mask=[False, True, True, False])

    @abstractproperty
    def _func(self):
        pass

    def test_strict_comparison(self):
        # Comparing both mask and data array completely.
        with self.assertRaises(AssertionError):
            self._func(self.arr1, self.arr2, strict=True)

    def test_no_strict_comparison(self):
        # Checking masked array equality and unmasked data values.
        self._func(self.arr1, self.arr2)

    def test_nomask(self):
        # Test that an assertion is raised when comparing missing mask with
        # mask containing True.
        arr1 = np.ma.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            self._func(arr1, self.arr2)

    def test_nomask_unmasked(self):
        # Ensure that a missing mask can compare with an unmasked array
        # object.
        arr1 = np.ma.array([1, 2, 3, 4])
        arr2 = np.ma.array([1, 2, 3, 4], mask=False)
        self._func(arr1, arr2)


class Test_assertMaskedArrayEqual(_MaskedArrayEquality, tests.IrisTest):
    @property
    def _func(self):
        return self.assertMaskedArrayEqual


class Test_assertMaskedArrayAlmostEqual(_MaskedArrayEquality, tests.IrisTest):
    @property
    def _func(self):
        return self.assertMaskedArrayAlmostEqual


if __name__ == '__main__':
    tests.main()
