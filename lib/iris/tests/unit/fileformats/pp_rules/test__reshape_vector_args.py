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
"""
Unit tests for
:func:`iris.fileformats.pp_rules._reshape_vector_args`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.fileformats.pp_rules import _reshape_vector_args


class TestEmpty(tests.IrisTest):
    def test(self):
        result = _reshape_vector_args([])
        self.assertEqual(result, [])


class TestSingleArg(tests.IrisTest):
    def _check_result(self, array):
        self.assertEqual(len(self.result), 1)
        self.assertArrayEqual(self.result[0], array)

    def test_nochange(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        self.result = _reshape_vector_args([(array, (0, 1))])
        self._check_result(array)

    def test_bad_dimensions(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegexp(ValueError, 'Length'):
            self.result = _reshape_vector_args([(array, (0, 1, 2))])

    def test_scalar(self):
        array = 5
        self.result = _reshape_vector_args([(array, ())])
        self._check_result(array)

    def test_nonarray(self):
        array = [[1, 2, 3], [4, 5, 6]]
        self.result = _reshape_vector_args([(array, (0, 1))])
        self._check_result(np.array(array))

    def test_transpose(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        self.result = _reshape_vector_args([(array, (1, 0))])
        self._check_result(array.T)

    def test_extend(self):
        array = np.array([[1, 2, 3, 4], [21, 22, 23, 24], [31, 32, 33, 34]])
        self.result = _reshape_vector_args([(array, (1, 3))])
        self._check_result(array.reshape((1, 3, 1, 4)))


class TestMultipleArgs(tests.IrisTest):
    def _check_results(self, arrays):
        self.assertEqual(len(self.result), len(arrays))
        for result, array in zip(self.result, arrays):
            self.assertArrayEqual(result, array)

    def test_nochange(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[0, 2, 4], [7, 8, 9]])
        self.result = _reshape_vector_args([(a1, (0, 1)), (a2, (0, 1))])
        self._check_results([a1, a2])

    def test_array_and_scalar(self):
        a1 = [[1, 2, 3], [3, 4, 5]]
        a2 = 5
        self.result = _reshape_vector_args([(a1, (0, 1)), (a2, ())])
        self._check_results([a1, np.array([[5]])])

    def test_transpose(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[0, 2, 4], [7, 8, 9]])
        self.result = _reshape_vector_args([(a1, (0, 1)), (a2, (1, 0))])
        self._check_results([a1, a2.T])

    def test_incompatible(self):
        # Does not enforce compatibility of results.
        a1 = np.array([1, 2])
        a2 = np.array([1, 2, 3])
        self.result = _reshape_vector_args([(a1, (0,)), (a2, (0,))])
        self._check_results([a1, a2])

    def test_extend(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([11, 12, 13])
        self.result = _reshape_vector_args([(a1, (0, 1)), (a2, (1,))])
        self._check_results([a1, a2.reshape((1, 3))])

    def test_extend_transpose(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([11, 12, 13])
        self.result = _reshape_vector_args([(a1, (1, 0)), (a2, (1,))])
        self._check_results([a1.T, a2.reshape((1, 3))])

    def test_double_extend(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array(1)
        self.result = _reshape_vector_args([(a1, (0, 2)), (a2, ())])
        self._check_results([a1.reshape(2, 1, 3), a2.reshape((1, 1, 1))])

    def test_triple(self):
        a1 = np.array([[1, 2, 3, 4]])
        a2 = np.array([3, 4])
        a3 = np.array(7)
        self.result = _reshape_vector_args(
            [(a1, (0, 2)), (a2, (1,)), (a3, ())])
        self._check_results([(a1.reshape(1, 1, 4)),
                             (a2.reshape(1, 2, 1)),
                             (a3.reshape(1, 1, 1))])

if __name__ == "__main__":
    tests.main()
