# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.array_equal`."""

import numpy as np
import numpy.ma as ma

from iris.util import array_equal


class Test:
    def test_0d(self):
        array_a = np.array(23)
        array_b = np.array(23)
        array_c = np.array(7)
        assert array_equal(array_a, array_b)
        assert not array_equal(array_a, array_c)

    def test_0d_and_scalar(self):
        array_a = np.array(23)
        assert array_equal(array_a, 23)
        assert not array_equal(array_a, 45)

    def test_1d_and_sequences(self):
        for sequence_type in (list, tuple):
            seq_a = sequence_type([1, 2, 3])
            array_a = np.array(seq_a)
            assert array_equal(array_a, seq_a)
            assert not array_equal(array_a, seq_a[:-1])
            array_a[1] = 45
            assert not array_equal(array_a, seq_a)

    def test_nd(self):
        array_a = np.array(np.arange(24).reshape(2, 3, 4))
        array_b = np.array(np.arange(24).reshape(2, 3, 4))
        array_c = np.array(np.arange(24).reshape(2, 3, 4))
        array_c[0, 1, 2] = 100
        assert array_equal(array_a, array_b)
        assert not array_equal(array_a, array_c)

    def test_masked_is_not_ignored(self):
        array_a = ma.masked_array([1, 2, 3], mask=[1, 0, 1])
        array_b = ma.masked_array([2, 2, 2], mask=[1, 0, 1])
        assert array_equal(array_a, array_b)

    def test_masked_is_different(self):
        array_a = ma.masked_array([1, 2, 3], mask=[1, 0, 1])
        array_b = ma.masked_array([1, 2, 3], mask=[0, 0, 1])
        assert not array_equal(array_a, array_b)

    def test_masked_isnt_unmasked(self):
        array_a = np.array([1, 2, 2])
        array_b = ma.masked_array([1, 2, 2], mask=[0, 0, 1])
        assert not array_equal(array_a, array_b)

    def test_masked_unmasked_equivelance(self):
        array_a = np.array([1, 2, 2])
        array_b = ma.masked_array([1, 2, 2])
        array_c = ma.masked_array([1, 2, 2], mask=[0, 0, 0])
        assert array_equal(array_a, array_b)
        assert array_equal(array_a, array_c)

    def test_fully_masked_arrays(self):
        array_a = ma.masked_array(np.arange(24).reshape(2, 3, 4), mask=True)
        array_b = ma.masked_array(np.arange(24).reshape(2, 3, 4), mask=True)
        assert array_equal(array_a, array_b)

    def test_fully_masked_0d_arrays(self):
        array_a = ma.masked_array(3, mask=True)
        array_b = ma.masked_array(3, mask=True)
        assert array_equal(array_a, array_b)

    def test_fully_masked_string_arrays(self):
        array_a = ma.masked_array(["a", "b", "c"], mask=True)
        array_b = ma.masked_array(["a", "b", "c"], mask=[1, 1, 1])
        assert array_equal(array_a, array_b)

    def test_partially_masked_string_arrays(self):
        array_a = ma.masked_array(["a", "b", "c"], mask=[1, 0, 1])
        array_b = ma.masked_array(["a", "b", "c"], mask=[1, 0, 1])
        assert array_equal(array_a, array_b)

    def test_string_arrays_equal(self):
        array_a = np.array(["abc", "def", "efg"])
        array_b = np.array(["abc", "def", "efg"])
        assert array_equal(array_a, array_b)

    def test_string_arrays_different_contents(self):
        array_a = np.array(["abc", "def", "efg"])
        array_b = np.array(["abc", "de", "efg"])
        assert not array_equal(array_a, array_b)

    def test_string_arrays_subset(self):
        array_a = np.array(["abc", "def", "efg"])
        array_b = np.array(["abc", "def"])
        assert not array_equal(array_a, array_b)
        assert not array_equal(array_b, array_a)

    def test_string_arrays_unequal_dimensionality(self):
        array_a = np.array("abc")
        array_b = np.array(["abc"])
        array_c = np.array([["abc"]])
        assert not array_equal(array_a, array_b)
        assert not array_equal(array_b, array_a)
        assert not array_equal(array_a, array_c)
        assert not array_equal(array_b, array_c)

    def test_string_arrays_0d_and_scalar(self):
        array_a = np.array("foobar")
        assert array_equal(array_a, "foobar")
        assert not array_equal(array_a, "foo")
        assert not array_equal(array_a, "foobar.")

    def test_nan_equality_nan_ne_nan(self):
        array_a = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        array_b = array_a.copy()
        assert not array_equal(array_a, array_a)
        assert not array_equal(array_a, array_b)

    def test_nan_equality_nan_naneq_nan(self):
        array_a = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        array_b = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        assert array_equal(array_a, array_a, withnans=True)
        assert array_equal(array_a, array_b, withnans=True)

    def test_nan_equality_nan_nanne_a(self):
        array_a = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        array_b = np.array([1.0, np.nan, 2.0, 0.0, 3.0])
        assert not array_equal(array_a, array_b, withnans=True)

    def test_nan_equality_a_nanne_b(self):
        array_a = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        array_b = np.array([1.0, np.nan, 2.0, np.nan, 4.0])
        assert not array_equal(array_a, array_b, withnans=True)
