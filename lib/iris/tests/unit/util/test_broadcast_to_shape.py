# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.broadcast_to_shape`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris.util import broadcast_to_shape


class Test_broadcast_to_shape(tests.IrisTest):
    def test_same_shape(self):
        # broadcast to current shape should result in no change
        a = np.random.random([2, 3])
        b = broadcast_to_shape(a, a.shape, (0, 1))
        self.assertArrayEqual(b, a)

    def test_added_dimensions(self):
        # adding two dimensions, on at the front and one in the middle of
        # the existing dimensions
        a = np.random.random([2, 3])
        b = broadcast_to_shape(a, (5, 2, 4, 3), (1, 3))
        for i in range(5):
            for j in range(4):
                self.assertArrayEqual(b[i, :, j, :], a)

    def test_added_dimensions_transpose(self):
        # adding dimensions and having the dimensions of the input
        # transposed
        a = np.random.random([2, 3])
        b = broadcast_to_shape(a, (5, 3, 4, 2), (3, 1))
        for i in range(5):
            for j in range(4):
                self.assertArrayEqual(b[i, :, j, :].T, a)

    def test_masked(self):
        # masked arrays are also accepted
        a = np.random.random([2, 3])
        m = ma.array(a, mask=[[0, 1, 0], [0, 1, 1]])
        b = broadcast_to_shape(m, (5, 3, 4, 2), (3, 1))
        for i in range(5):
            for j in range(4):
                self.assertMaskedArrayEqual(b[i, :, j, :].T, m)

    def test_masked_degenerate(self):
        # masked arrays can have degenerate masks too
        a = np.random.random([2, 3])
        m = ma.array(a)
        b = broadcast_to_shape(m, (5, 3, 4, 2), (3, 1))
        for i in range(5):
            for j in range(4):
                self.assertMaskedArrayEqual(b[i, :, j, :].T, m)


if __name__ == "__main__":
    tests.main()
