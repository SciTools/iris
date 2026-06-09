# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformat.ff.ENDGame`."""

import numpy as np

from iris.fileformats._ff import ENDGame
from iris.tests import _shared_utils


class Test:
    def test_class_attributes(self):
        reals = np.arange(6) + 100
        grid = ENDGame(None, None, reals, None)
        assert grid._v_offset == -0.5


class Test__y_vectors:
    def _test(self, row, yp, yv):
        reals = np.arange(6) + 100
        grid = ENDGame(None, row, reals, None)
        result_yp, result_yv = grid._y_vectors()
        _shared_utils.assert_array_equal(result_yp, yp)
        _shared_utils.assert_array_equal(result_yv, yv)

    def test_none(self):
        self._test(row=None, yp=None, yv=None)

    def test_1d(self):
        self._test(row=np.array([[0], [1], [2], [3]]), yp=np.array([0, 1, 2]), yv=None)

    def test_2d(self):
        self._test(
            row=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
            yp=np.array([0, 1, 2]),
            yv=np.array([0, 10, 20, 30]),
        )
