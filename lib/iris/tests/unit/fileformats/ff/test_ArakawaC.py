# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformat.ff.ArakawaC`."""

import numpy as np

from iris.fileformats._ff import ArakawaC
from iris.tests import _shared_utils


class Test__x_vectors:
    def _test(self, column, horiz_grid_type, xp, xu):
        reals = np.arange(6) + 100
        grid = ArakawaC(column, None, reals, horiz_grid_type)
        result_xp, result_xu = grid._x_vectors()
        _shared_utils.assert_array_equal(result_xp, xp)
        _shared_utils.assert_array_equal(result_xu, xu)

    def test_none(self):
        self._test(column=None, horiz_grid_type=None, xp=None, xu=None)

    def test_1d(self):
        self._test(
            column=np.array([[0], [1], [2], [3]]),
            horiz_grid_type=None,
            xp=np.array([0, 1, 2, 3]),
            xu=None,
        )

    def test_2d_no_wrap(self):
        self._test(
            column=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
            horiz_grid_type=1,
            xp=np.array([0, 1, 2, 3]),
            xu=np.array([0, 10, 20, 30]),
        )

    def test_2d_with_wrap(self):
        self._test(
            column=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
            horiz_grid_type=0,
            xp=np.array([0, 1, 2, 3]),
            xu=np.array([0, 10, 20]),
        )


class Test_regular_x:
    def _test(self, subgrid, bzx, bdx):
        grid = ArakawaC(None, None, [4.0, None, None, -5.0, None, None], None)
        result_bzx, result_bdx = grid.regular_x(subgrid)
        assert result_bzx == bzx
        assert result_bdx == bdx

    def test_theta_subgrid(self):
        self._test(1, -9.0, 4.0)

    def test_u_subgrid(self):
        self._test(11, -7.0, 4.0)


class Test_regular_y:
    def _test(self, v_offset, subgrid, bzy, bdy):
        grid = ArakawaC(None, None, [None, 4.0, 45.0, None, None, None], None)
        grid._v_offset = v_offset
        result_bzy, result_bdy = grid.regular_y(subgrid)
        assert result_bzy == bzy
        assert result_bdy == bdy

    def test_theta_subgrid_NewDynamics(self):
        self._test(0.5, 1, 41.0, 4.0)

    def test_v_subgrid_NewDynamics(self):
        self._test(0.5, 11, 43.0, 4.0)

    def test_theta_subgrid_ENDGame(self):
        self._test(-0.5, 1, 41.0, 4.0)

    def test_v_subgrid_ENDGame(self):
        self._test(-0.5, 11, 39.0, 4.0)
