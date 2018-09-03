# (C) British Crown Copyright 2010 - 2018, Met Office
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
"""Test function :func:`iris.util.reverse`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest

import iris
import numpy as np


class Test(tests.IrisTest):
    def SetUpReversed(self):
        # On this cube pair, the coordinates to perform operations on have
        # matching long names but the points array on one cube is reversed
        # with respect to that on the other.
        data = np.zeros((3, 4))
        a1 = iris.coords.DimCoord([1, 2, 3], long_name='a')
        b1 = iris.coords.DimCoord([1, 2, 3, 4], long_name='b')
        a2 = iris.coords.DimCoord([3, 2, 1], long_name='a')
        b2 = iris.coords.DimCoord([4, 3, 2, 1], long_name='b')

        reversed1 = iris.cube.Cube(
            data, dim_coords_and_dims=[(a1, 0), (b1, 1)])
        reversed2 = iris.cube.Cube(
            data, dim_coords_and_dims=[(a2, 0), (b2, 1)])

        return reversed1, reversed2

    def test_simple_array(self):
        a = np.arange(12).reshape(3, 4)
        self.assertArrayEqual(a[::-1], iris.util.reverse(a, 0))
        self.assertArrayEqual(a[::-1, ::-1], iris.util.reverse(a, [0, 1]))
        self.assertArrayEqual(a[:, ::-1], iris.util.reverse(a, 1))
        self.assertArrayEqual(a[:, ::-1], iris.util.reverse(a, [1]))
        self.assertRaises(ValueError, iris.util.reverse, a, [])
        self.assertRaises(ValueError, iris.util.reverse, a, -1)
        self.assertRaises(ValueError, iris.util.reverse, a, 10)
        self.assertRaises(ValueError, iris.util.reverse, a, [-1])
        self.assertRaises(ValueError, iris.util.reverse, a, [0, -1])

    def test_single_array(self):
        a = np.arange(36).reshape(3, 4, 3)
        self.assertArrayEqual(a[::-1], iris.util.reverse(a, 0))
        self.assertArrayEqual(a[::-1, ::-1], iris.util.reverse(a, [0, 1]))
        self.assertArrayEqual(a[:, ::-1, ::-1], iris.util.reverse(a, [1, 2]))
        self.assertArrayEqual(a[..., ::-1], iris.util.reverse(a, 2))
        self.assertRaises(ValueError, iris.util.reverse, a, -1)
        self.assertRaises(ValueError, iris.util.reverse, a, 10)
        self.assertRaises(ValueError, iris.util.reverse, a, [-1])
        self.assertRaises(ValueError, iris.util.reverse, a, [0, -1])

    def test_cube(self):
        cube1, cube2 = self.SetUpReversed()

        cube1_reverse0 = iris.util.reverse(cube1, 0)
        cube1_reverse1 = iris.util.reverse(cube1, 1)
        cube1_reverse_both = iris.util.reverse(cube1, (0, 1))

        self.assertArrayEqual(cube1.data[::-1], cube1_reverse0.data)
        self.assertArrayEqual(cube2.coord('a').points,
                              cube1_reverse0.coord('a').points)
        self.assertArrayEqual(cube1.coord('b').points,
                              cube1_reverse0.coord('b').points)

        self.assertArrayEqual(cube1.data[:, ::-1], cube1_reverse1.data)
        self.assertArrayEqual(cube1.coord('a').points,
                              cube1_reverse1.coord('a').points)
        self.assertArrayEqual(cube2.coord('b').points,
                              cube1_reverse1.coord('b').points)

        self.assertArrayEqual(cube1.data[::-1, ::-1], cube1_reverse_both.data)
        self.assertArrayEqual(cube2.coord('a').points,
                              cube1_reverse_both.coord('a').points)
        self.assertArrayEqual(cube2.coord('b').points,
                              cube1_reverse_both.coord('b').points)


if __name__ == '__main__':
    unittest.main()
