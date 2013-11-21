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
"""Unit tests for :class:`iris.fileformat.ff.ArakawaC`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.fileformats.ff import ArakawaC


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
        self._test(column=np.array([[0], [1], [2], [3]]),
                   horiz_grid_type=None,
                   xp=np.array([0, 1, 2, 3]), xu=None)

    def test_2d_no_wrap(self):
        self._test(column=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
                   horiz_grid_type=1,
                   xp=np.array([0, 1, 2, 3]),
                   xu=np.array([0, 10, 20, 30]))

    def test_2d_with_wrap(self):
        self._test(column=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
                   horiz_grid_type=0,
                   xp=np.array([0, 1, 2, 3]),
                   xu=np.array([0, 10, 20]))


if __name__ == "__main__":
    tests.main()
