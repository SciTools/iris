# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris.fileformat.ff.NewDynamics`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.fileformats._ff import NewDynamics


class Test(tests.IrisTest):
    def test_class_attributes(self):
        reals = np.arange(6) + 100
        grid = NewDynamics(None, None, reals, None)
        self.assertEqual(grid._v_offset, 0.5)


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
        self._test(
            row=np.array([[0], [1], [2], [3]]),
            yp=np.array([0, 1, 2, 3]),
            yv=None,
        )

    def test_2d(self):
        self._test(
            row=np.array([[0, 0], [1, 10], [2, 20], [3, 30]]),
            yp=np.array([0, 1, 2, 3]),
            yv=np.array([0, 10, 20]),
        )


if __name__ == "__main__":
    tests.main()
