# (C) British Crown Copyright 2018, Met Office
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
from iris.util import reverse
import numpy as np


class Test_array(tests.IrisTest):
    def test_simple_array(self):
        a = np.arange(12).reshape(3, 4)
        self.assertArrayEqual(a[::-1], reverse(a, 0))
        self.assertArrayEqual(a[::-1, ::-1], reverse(a, [0, 1]))
        self.assertArrayEqual(a[:, ::-1], reverse(a, 1))
        self.assertArrayEqual(a[:, ::-1], reverse(a, [1]))

        msg = 'Reverse was expecting a single axis or a 1d array *'
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, [])

        msg = 'An axis value out of range for the number of dimensions *'
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, -1)
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, 10)
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, [-1])
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, [0, -1])

        msg = 'To reverse an array, provide an int *'
        with self.assertRaisesRegexp(TypeError, msg):
            reverse(a, 'latitude')

    def test_single_array(self):
        a = np.arange(36).reshape(3, 4, 3)
        self.assertArrayEqual(a[::-1], reverse(a, 0))
        self.assertArrayEqual(a[::-1, ::-1], reverse(a, [0, 1]))
        self.assertArrayEqual(a[:, ::-1, ::-1], reverse(a, [1, 2]))
        self.assertArrayEqual(a[..., ::-1], reverse(a, 2))

        msg = 'Reverse was expecting a single axis or a 1d array *'
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, [])

        msg = 'An axis value out of range for the number of dimensions *'
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, -1)
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, 10)
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, [-1])
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(a, [0, -1])

        with self.assertRaisesRegexp(
                TypeError, 'To reverse an array, provide an int *'):
            reverse(a, 'latitude')


class Test_cube(tests.IrisTest):
    def setUp(self):
        # On this cube pair, the coordinates to perform operations on have
        # matching long names but the points array on one cube is reversed
        # with respect to that on the other.
        data = np.arange(12).reshape(3, 4)
        self.a1 = iris.coords.DimCoord([1, 2, 3], long_name='a')
        self.b1 = iris.coords.DimCoord([1, 2, 3, 4], long_name='b')
        a2 = iris.coords.DimCoord([3, 2, 1], long_name='a')
        b2 = iris.coords.DimCoord([4, 3, 2, 1], long_name='b')
        self.span = iris.coords.AuxCoord(np.arange(12).reshape(3, 4),
                                         long_name='spanning')

        self.cube1 = iris.cube.Cube(
            data, dim_coords_and_dims=[(self.a1, 0), (self.b1, 1)],
            aux_coords_and_dims=[(self.span, (0, 1))])

        self.cube2 = iris.cube.Cube(
            data, dim_coords_and_dims=[(a2, 0), (b2, 1)])

    def test_cube_dim(self):
        cube1_reverse0 = reverse(self.cube1, 0)
        cube1_reverse1 = reverse(self.cube1, 1)
        cube1_reverse_both = reverse(self.cube1, (0, 1))

        self.assertArrayEqual(self.cube1.data[::-1], cube1_reverse0.data)
        self.assertArrayEqual(self.cube2.coord('a').points,
                              cube1_reverse0.coord('a').points)
        self.assertArrayEqual(self.cube1.coord('b').points,
                              cube1_reverse0.coord('b').points)

        self.assertArrayEqual(self.cube1.data[:, ::-1], cube1_reverse1.data)
        self.assertArrayEqual(self.cube1.coord('a').points,
                              cube1_reverse1.coord('a').points)
        self.assertArrayEqual(self.cube2.coord('b').points,
                              cube1_reverse1.coord('b').points)

        self.assertArrayEqual(self.cube1.data[::-1, ::-1],
                              cube1_reverse_both.data)
        self.assertArrayEqual(self.cube2.coord('a').points,
                              cube1_reverse_both.coord('a').points)
        self.assertArrayEqual(self.cube2.coord('b').points,
                              cube1_reverse_both.coord('b').points)

    def test_cube_coord(self):
        cube1_reverse0 = reverse(self.cube1, self.a1)
        cube1_reverse1 = reverse(self.cube1, 'b')
        cube1_reverse_both = reverse(self.cube1, (self.a1, self.b1))
        cube1_reverse_spanning = reverse(self.cube1, 'spanning')

        self.assertArrayEqual(self.cube1.data[::-1], cube1_reverse0.data)
        self.assertArrayEqual(self.cube2.coord('a').points,
                              cube1_reverse0.coord('a').points)
        self.assertArrayEqual(self.cube1.coord('b').points,
                              cube1_reverse0.coord('b').points)

        self.assertArrayEqual(self.cube1.data[:, ::-1], cube1_reverse1.data)
        self.assertArrayEqual(self.cube1.coord('a').points,
                              cube1_reverse1.coord('a').points)
        self.assertArrayEqual(self.cube2.coord('b').points,
                              cube1_reverse1.coord('b').points)

        self.assertArrayEqual(self.cube1.data[::-1, ::-1],
                              cube1_reverse_both.data)
        self.assertArrayEqual(self.cube2.coord('a').points,
                              cube1_reverse_both.coord('a').points)
        self.assertArrayEqual(self.cube2.coord('b').points,
                              cube1_reverse_both.coord('b').points)

        self.assertArrayEqual(self.cube1.data[::-1, ::-1],
                              cube1_reverse_spanning.data)
        self.assertArrayEqual(self.cube2.coord('a').points,
                              cube1_reverse_spanning.coord('a').points)
        self.assertArrayEqual(self.cube2.coord('b').points,
                              cube1_reverse_spanning.coord('b').points)
        self.assertArrayEqual(
            self.span.points[::-1, ::-1],
            cube1_reverse_spanning.coord('spanning').points)

        msg = 'Expected to find exactly 1 latitude coordinate, but found none.'
        with self.assertRaisesRegexp(
                iris.exceptions.CoordinateNotFoundError, msg):
            reverse(self.cube1, 'latitude')

        msg = 'Reverse was expecting a single axis or a 1d array *'
        with self.assertRaisesRegexp(ValueError, msg):
            reverse(self.cube1, [])

        msg = ('coords_or_dims must be int, str, coordinate or sequence of '
               'these.  Got cube.')
        with self.assertRaisesRegexp(TypeError, msg):
            reverse(self.cube1, self.cube1)

        msg = ('coords_or_dims must be int, str, coordinate or sequence of '
               'these.')
        with self.assertRaisesRegexp(TypeError, msg):
            reverse(self.cube1, 3.)


if __name__ == '__main__':
    unittest.main()
