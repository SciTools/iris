# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for
:func:`iris.fileformats.pp_load_rules._reshape_vector_args`.

"""

import numpy as np
import pytest

from iris.fileformats.pp_load_rules import _reshape_vector_args
from iris.tests._shared_utils import assert_array_equal


class TestEmpty:
    def test(self):
        result = _reshape_vector_args([])
        assert result == []


class TestSingleArg:
    def _check(self, result, expected):
        assert len(result) == len(expected)
        for result_arr, expected_arr in zip(result, expected):
            assert_array_equal(result_arr, expected_arr)

    def test_nochange(self):
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = _reshape_vector_args([(points, (0, 1))])
        expected = [points]
        self._check(result, expected)

    def test_bad_dimensions(self):
        points = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="Length"):
            _reshape_vector_args([(points, (0, 1, 2))])

    def test_scalar(self):
        points = 5
        result = _reshape_vector_args([(points, ())])
        expected = [points]
        self._check(result, expected)

    def test_nonarray(self):
        points = [[1, 2, 3], [4, 5, 6]]
        result = _reshape_vector_args([(points, (0, 1))])
        expected = [np.array(points)]
        self._check(result, expected)

    def test_transpose(self):
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = _reshape_vector_args([(points, (1, 0))])
        expected = [points.T]
        self._check(result, expected)

    def test_extend(self):
        points = np.array([[1, 2, 3, 4], [21, 22, 23, 24], [31, 32, 33, 34]])
        result = _reshape_vector_args([(points, (1, 3))])
        expected = [points.reshape(1, 3, 1, 4)]
        self._check(result, expected)


class TestMultipleArgs:
    def _check(self, result, expected):
        assert len(result) == len(expected)
        for result_arr, expected_arr in zip(result, expected):
            assert_array_equal(result_arr, expected_arr)

    def test_nochange(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[0, 2, 4], [7, 8, 9]])
        result = _reshape_vector_args([(a1, (0, 1)), (a2, (0, 1))])
        expected = [a1, a2]
        self._check(result, expected)

    def test_array_and_scalar(self):
        a1 = [[1, 2, 3], [3, 4, 5]]
        a2 = 5
        result = _reshape_vector_args([(a1, (0, 1)), (a2, ())])
        expected = [a1, np.array([[5]])]
        self._check(result, expected)

    def test_transpose(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[0, 2, 4], [7, 8, 9]])
        result = _reshape_vector_args([(a1, (0, 1)), (a2, (1, 0))])
        expected = [a1, a2.T]
        self._check(result, expected)

    def test_incompatible(self):
        # Does not enforce compatibility of results.
        a1 = np.array([1, 2])
        a2 = np.array([1, 2, 3])
        result = _reshape_vector_args([(a1, (0,)), (a2, (0,))])
        expected = [a1, a2]
        self._check(result, expected)

    def test_extend(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([11, 12, 13])
        result = _reshape_vector_args([(a1, (0, 1)), (a2, (1,))])
        expected = [a1, a2.reshape(1, 3)]
        self._check(result, expected)

    def test_extend_transpose(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([11, 12, 13])
        result = _reshape_vector_args([(a1, (1, 0)), (a2, (1,))])
        expected = [a1.T, a2.reshape(1, 3)]
        self._check(result, expected)

    def test_double_extend(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array(1)
        result = _reshape_vector_args([(a1, (0, 2)), (a2, ())])
        expected = [a1.reshape(2, 1, 3), a2.reshape(1, 1, 1)]
        self._check(result, expected)

    def test_triple(self):
        a1 = np.array([[1, 2, 3, 4]])
        a2 = np.array([3, 4])
        a3 = np.array(7)
        result = _reshape_vector_args([(a1, (0, 2)), (a2, (1,)), (a3, ())])
        expected = [
            a1.reshape(1, 1, 4),
            a2.reshape(1, 2, 1),
            a3.reshape(1, 1, 1),
        ]
        self._check(result, expected)
