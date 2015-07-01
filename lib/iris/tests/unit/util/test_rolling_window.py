# (C) British Crown Copyright 2013 - 2015, Met Office
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
"""Test function :func:`iris.util.rolling_window`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.util import rolling_window


class Test_rolling_window(tests.IrisTest):

    def test_1d(self):
        # 1-d array input
        a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        expected_result = np.array([[0, 1], [1, 2], [2, 3], [3, 4]],
                                   dtype=np.int32)
        result = rolling_window(a, window=2)
        self.assertArrayEqual(result, expected_result)

    def test_2d(self):
        # 2-d array input
        a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32)
        expected_result = np.array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                                    [[5, 6, 7], [6, 7, 8], [7, 8, 9]]],
                                   dtype=np.int32)
        result = rolling_window(a, window=3, axis=1)
        self.assertArrayEqual(result, expected_result)

    def test_1d_masked(self):
        # 1-d masked array input
        a = ma.array([0, 1, 2, 3, 4], mask=[0, 0, 1, 0, 0],
                     dtype=np.int32)
        expected_result = ma.array([[0, 1], [1, 2], [2, 3], [3, 4]],
                                   mask=[[0, 0], [0, 1], [1, 0], [0, 0]],
                                   dtype=np.int32)
        result = rolling_window(a, window=2)
        self.assertMaskedArrayEqual(result, expected_result)

    def test_2d_masked(self):
        # 2-d masked array input
        a = ma.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                     mask=[[0, 0, 1, 0, 0], [1, 0, 1, 0, 0]],
                     dtype=np.int32)
        expected_result = ma.array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                                    [[5, 6, 7], [6, 7, 8], [7, 8, 9]]],
                                   mask=[[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                                         [[1, 0, 1], [0, 1, 0], [1, 0, 0]]],
                                   dtype=np.int32)
        result = rolling_window(a, window=3, axis=1)
        self.assertMaskedArrayEqual(result, expected_result)

    def test_degenerate_mask(self):
        a = ma.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32)
        expected_result = ma.array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                                    [[5, 6, 7], [6, 7, 8], [7, 8, 9]]],
                                   mask=[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                   dtype=np.int32)
        result = rolling_window(a, window=3, axis=1)
        self.assertMaskedArrayEqual(result, expected_result)

    def test_step(self):
        # step should control how far apart consecutive windows are
        a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32)
        expected_result = np.array([[[0, 1, 2], [2, 3, 4]],
                                    [[5, 6, 7], [7, 8, 9]]],
                                   dtype=np.int32)
        result = rolling_window(a, window=3, step=2, axis=1)
        self.assertArrayEqual(result, expected_result)

    def test_window_too_short(self):
        # raise an error if the window length is less than 1
        a = np.empty([5])
        with self.assertRaises(ValueError):
            rolling_window(a, window=0)

    def test_window_too_long(self):
        # raise an error if the window length is longer than the
        # corresponding array dimension
        a = np.empty([7, 5])
        with self.assertRaises(ValueError):
            rolling_window(a, window=6, axis=1)

    def test_invalid_step(self):
        # raise an error if the step between windows is less than 1
        a = np.empty([5])
        with self.assertRaises(ValueError):
            rolling_window(a, step=0)


if __name__ == '__main__':
    tests.main()
