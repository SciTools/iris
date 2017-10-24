# (C) British Crown Copyright 2014 - 2017, Met Office
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
"""
Unit tests for
:func:`iris.fileformats.pp_load_rules._reshape_vector_args`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.fileformats.pp_load_rules import _reshape_vector_args


class TestEmpty(tests.IrisTest):
    def test(self):
        result = _reshape_vector_args([])
        self.assertEqual(result, [])


class TestSingleArg(tests.IrisTest):
    def _check(self, result, expected):
        self.assertEqual(len(result), len(expected))
        for result_arr, expected_arr in zip(result, expected):
            self.assertArrayEqual(result_arr, expected_arr)

    def test_nochange(self):
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = _reshape_vector_args([(points, (0, 1))])
        expected = [points]
        self._check(result, expected)

    def test_bad_dimensions(self):
        points = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegexp(ValueError, 'Length'):
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


class TestMultipleArgs(tests.IrisTest):
    def _check(self, result, expected):
        self.assertEqual(len(result), len(expected))
        for result_arr, expected_arr in zip(result, expected):
            self.assertArrayEqual(result_arr, expected_arr)

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
        expected = [a1.reshape(1, 1, 4),
                    a2.reshape(1, 2, 1),
                    a3.reshape(1, 1, 1)]
        self._check(result, expected)


if __name__ == "__main__":
    tests.main()
