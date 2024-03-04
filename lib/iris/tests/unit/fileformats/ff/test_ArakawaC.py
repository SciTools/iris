# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris.fileformat.ff.ArakawaC`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.fileformats._ff import ArakawaC


class Test__x_vectors(tests.IrisTest):
    def _test(self, column, horiz_grid_type, xp, xu):
        reals = np.arange(6) + 100
        grid = ArakawaC(column, None, reals, horiz_grid_type)
        result_xp, result_xu = grid._x_vectors()
        self.assertArrayEqual(result_xp, xp)
        self.assertArrayEqual(result_xu, xu)

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


class Test_regular_x(tests.IrisTest):
    def _test(self, subgrid, bzx, bdx):
        grid = ArakawaC(None, None, [4.0, None, None, -5.0, None, None], None)
        result_bzx, result_bdx = grid.regular_x(subgrid)
        self.assertEqual(result_bzx, bzx)
        self.assertEqual(result_bdx, bdx)

    def test_theta_subgrid(self):
        self._test(1, -9.0, 4.0)

    def test_u_subgrid(self):
        self._test(11, -7.0, 4.0)


class Test_regular_y(tests.IrisTest):
    def _test(self, v_offset, subgrid, bzy, bdy):
        grid = ArakawaC(None, None, [None, 4.0, 45.0, None, None, None], None)
        grid._v_offset = v_offset
        result_bzy, result_bdy = grid.regular_y(subgrid)
        self.assertEqual(result_bzy, bzy)
        self.assertEqual(result_bdy, bdy)

    def test_theta_subgrid_NewDynamics(self):
        self._test(0.5, 1, 41.0, 4.0)

    def test_v_subgrid_NewDynamics(self):
        self._test(0.5, 11, 43.0, 4.0)

    def test_theta_subgrid_ENDGame(self):
        self._test(-0.5, 1, 41.0, 4.0)

    def test_v_subgrid_ENDGame(self):
        self._test(-0.5, 11, 39.0, 4.0)


if __name__ == "__main__":
    tests.main()
