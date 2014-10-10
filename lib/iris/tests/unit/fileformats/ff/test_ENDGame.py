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
"""Unit tests for :class:`iris.fileformat.ff.ENDGame`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.fileformats.ff import ENDGame


class Test(tests.IrisTest):
    def test_class_attributes(self):
        reals = np.arange(6) + 100
        grid = ENDGame(None, None, reals, None)
        self.assertEqual(grid._v_offset, -0.5)


class Test__y_vectors(tests.IrisTest):
    def _test(self, row, yp, yv):
        reals = np.arange(6) + 100
        grid = ENDGame(None, row, reals, None)
        result_yp, result_yv = grid._y_vectors()
        self.assertArrayEqual(result_yp, yp)
        self.assertArrayEqual(result_yv, yv)

    def test_none(self):
        self._test(row=None, yp=None, yv=None)

    def test_1d(self):
        self._test(row=np.array([[0], [1], [2], [3]]),
                   yp=np.array([0, 1, 2]), yv=None)

    def test_2d(self):
        self._test(row=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
                   yp=np.array([0, 1, 2]), yv=np.array([0, 10, 20, 30]))


if __name__ == "__main__":
    tests.main()
