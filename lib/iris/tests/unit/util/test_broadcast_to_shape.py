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
"""Test function :func:`iris.util.broadcast_to_shape`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

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


if __name__ == '__main__':
    tests.main()
