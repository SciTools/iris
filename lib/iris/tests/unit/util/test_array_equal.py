# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Test function :func:`iris.util.array_equal`."""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.util import array_equal


class Test(tests.IrisTest):
    def test_0d(self):
        array_a = np.array(23)
        array_b = np.array(23)
        array_c = np.array(7)
        self.assertTrue(array_equal(array_a, array_b))
        self.assertFalse(array_equal(array_a, array_c))

    def test_0d_and_scalar(self):
        array_a = np.array(23)
        self.assertTrue(array_equal(array_a, 23))
        self.assertFalse(array_equal(array_a, 45))

    def test_1d_and_sequences(self):
        for sequence_type in (list, tuple):
            seq_a = sequence_type([1, 2, 3])
            array_a = np.array(seq_a)
            self.assertTrue(array_equal(array_a, seq_a))
            self.assertFalse(array_equal(array_a, seq_a[:-1]))
            array_a[1] = 45
            self.assertFalse(array_equal(array_a, seq_a))

    def test_nd(self):
        array_a = np.array(np.arange(24).reshape(2, 3, 4))
        array_b = np.array(np.arange(24).reshape(2, 3, 4))
        array_c = np.array(np.arange(24).reshape(2, 3, 4))
        array_c[0, 1, 2] = 100
        self.assertTrue(array_equal(array_a, array_b))
        self.assertFalse(array_equal(array_a, array_c))

    def test_masked_is_ignored(self):
        array_a = ma.masked_array([1, 2, 3], mask=[1, 0, 1])
        array_b = ma.masked_array([2, 2, 2], mask=[1, 0, 1])
        self.assertFalse(array_equal(array_a, array_b))

    def test_fully_masked_arrays(self):
        array_a = ma.masked_array(np.arange(24).reshape(2, 3, 4), mask=True)
        array_b = ma.masked_array(np.arange(24).reshape(2, 3, 4), mask=True)
        self.assertTrue(array_equal(array_a, array_b))

    def test_fully_masked_0d_arrays(self):
        array_a = ma.masked_array(3, mask=True)
        array_b = ma.masked_array(3, mask=True)
        self.assertTrue(array_equal(array_a, array_b))

    def test_fully_masked_string_arrays(self):
        array_a = ma.masked_array(['a', 'b', 'c'], mask=True)
        array_b = ma.masked_array(['a', 'b', 'c'], mask=[1, 1, 1])
        self.assertTrue(array_equal(array_a, array_b))

    def test_partially_masked_string_arrays(self):
        array_a = ma.masked_array(['a', 'b', 'c'], mask=[1, 0, 1])
        array_b = ma.masked_array(['a', 'b', 'c'], mask=[1, 0, 1])
        self.assertTrue(array_equal(array_a, array_b))

    def test_string_arrays_equal(self):
        array_a = np.array(['abc', 'def', 'efg'])
        array_b = np.array(['abc', 'def', 'efg'])
        self.assertTrue(array_equal(array_a, array_b))

    def test_string_arrays_different_contents(self):
        array_a = np.array(['abc', 'def', 'efg'])
        array_b = np.array(['abc', 'de', 'efg'])
        self.assertFalse(array_equal(array_a, array_b))

    def test_string_arrays_subset(self):
        array_a = np.array(['abc', 'def', 'efg'])
        array_b = np.array(['abc', 'def'])
        self.assertFalse(array_equal(array_a, array_b))
        self.assertFalse(array_equal(array_b, array_a))

    def test_string_arrays_unequal_dimensionality(self):
        array_a = np.array('abc')
        array_b = np.array(['abc'])
        array_c = np.array([['abc']])
        self.assertFalse(array_equal(array_a, array_b))
        self.assertFalse(array_equal(array_b, array_a))
        self.assertFalse(array_equal(array_a, array_c))
        self.assertFalse(array_equal(array_b, array_c))

    def test_string_arrays_0d_and_scalar(self):
        array_a = np.array('foobar')
        self.assertTrue(array_equal(array_a, 'foobar'))
        self.assertFalse(array_equal(array_a, 'foo'))
        self.assertFalse(array_equal(array_a, 'foobar.'))


if __name__ == '__main__':
    tests.main()
