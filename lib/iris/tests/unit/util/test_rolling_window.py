# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.rolling_window`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris.util import rolling_window


class Test_rolling_window(tests.IrisTest):
    def test_1d(self):
        # 1-d array input
        a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        expected_result = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
        result = rolling_window(a, window=2)
        self.assertArrayEqual(result, expected_result)

    def test_2d(self):
        # 2-d array input
        a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32)
        expected_result = np.array(
            [
                [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                [[5, 6, 7], [6, 7, 8], [7, 8, 9]],
            ],
            dtype=np.int32,
        )
        result = rolling_window(a, window=3, axis=1)
        self.assertArrayEqual(result, expected_result)

    def test_3d_lazy(self):
        a = da.arange(2 * 3 * 4).reshape((2, 3, 4))
        expected_result = np.arange(2 * 3 * 4).reshape((1, 2, 3, 4))
        result = rolling_window(a, window=2, axis=0).compute()
        self.assertArrayEqual(result, expected_result)

    def test_1d_masked(self):
        # 1-d masked array input
        a = ma.array([0, 1, 2, 3, 4], mask=[0, 0, 1, 0, 0], dtype=np.int32)
        expected_result = ma.array(
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            mask=[[0, 0], [0, 1], [1, 0], [0, 0]],
            dtype=np.int32,
        )
        result = rolling_window(a, window=2)
        self.assertMaskedArrayEqual(result, expected_result)

    def test_2d_masked(self):
        # 2-d masked array input
        a = ma.array(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            mask=[[0, 0, 1, 0, 0], [1, 0, 1, 0, 0]],
            dtype=np.int32,
        )
        expected_result = ma.array(
            [
                [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                [[5, 6, 7], [6, 7, 8], [7, 8, 9]],
            ],
            mask=[
                [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                [[1, 0, 1], [0, 1, 0], [1, 0, 0]],
            ],
            dtype=np.int32,
        )
        result = rolling_window(a, window=3, axis=1)
        self.assertMaskedArrayEqual(result, expected_result)

    def test_degenerate_mask(self):
        a = ma.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32)
        expected_result = ma.array(
            [
                [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                [[5, 6, 7], [6, 7, 8], [7, 8, 9]],
            ],
            mask=[
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=np.int32,
        )
        result = rolling_window(a, window=3, axis=1)
        self.assertMaskedArrayEqual(result, expected_result)

    def test_step(self):
        # step should control how far apart consecutive windows are
        a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32)
        expected_result = np.array(
            [[[0, 1, 2], [2, 3, 4]], [[5, 6, 7], [7, 8, 9]]], dtype=np.int32
        )
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


if __name__ == "__main__":
    tests.main()
