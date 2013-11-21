# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for :class:`iris.fileformat.ff.NewDynamics`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.fileformats.ff import NewDynamics


class Test__y_vectors(tests.IrisTest):
    def _test(self, row, yp, yv):
        reals = np.arange(6) + 100
        grid = NewDynamics(None, row, reals, None)
        result_yp, result_yv = grid._y_vectors()
        self.assertArrayEqual(result_yp, yp)
        self.assertArrayEqual(result_yv, yv)

    def test_none(self):
        self._test(row=None, yp=None, yv=None)

    def test_1d(self):
        self._test(row=np.array([[0], [1], [2], [3]]),
                   yp=np.array([0, 1, 2, 3]), yv=None)

    def test_2d(self):
        self._test(row=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
                   yp=np.array([0, 1, 2, 3]), yv=np.array([0, 10, 20]))


class Test_regular_x(tests.IrisTest):
    def _test(self, subgrid, bzx, bdx):
        reals = [4.0, None, None, -5.0, None, None]
        grid = NewDynamics(None, None, reals, None)
        result_bzx, result_bdx = grid.regular_x(subgrid)
        self.assertEqual(result_bzx, bzx)
        self.assertEqual(result_bdx, bdx)

    def test_theta_subgrid(self):
        self._test(1, -9.0, 4.0)

    def test_u_subgrid(self):
        self._test(11, -7.0, 4.0)


class Test_regular_y(tests.IrisTest):
    def _test(self, subgrid, bzy, bdy):
        reals = [None, 4.0, 45.0, None, None, None]
        grid = NewDynamics(None, None, reals, None)
        result_bzy, result_bdy = grid.regular_y(subgrid)
        self.assertEqual(result_bzy, bzy)
        self.assertEqual(result_bdy, bdy)

    def test_theta_subgrid(self):
        self._test(1, 41.0, 4.0)

    def test_v_subgrid(self):
        self._test(11, 43.0, 4.0)


if __name__ == "__main__":
    tests.main()
