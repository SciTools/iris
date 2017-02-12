# (C) British Crown Copyright 2013 - 2017, Met Office
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
"""Test function :func:`iris.util.move_dimension`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import unittest

import iris
import iris.tests.stock as stock
from iris.util import move_dimension


class Test(tests.IrisTest):
    def test_with_simple_1d_cube(self):
        cube = stock.simple_1d()
        move_dimension(cube, 'foo', position=0)
        self.assertEqual(cube.shape, (11,))

    def test_axis_already_in_position(self):
        cube = stock.simple_3d()
        move_dimension(cube, 'wibble', 0)
        self.assertEqual(cube.shape, (2, 3, 4))

        cube = stock.simple_3d()
        move_dimension(cube, 'wibble')
        self.assertEqual(cube.shape, (2, 3, 4))

        cube = stock.simple_3d()
        move_dimension(cube, 'wibble', -3)
        self.assertEqual(cube.shape, (2, 3, 4))

        cube = stock.simple_3d()
        move_dimension(cube, cube.coord('latitude'), 1)
        self.assertEqual(cube.shape, (2, 3, 4))

        cube = stock.simple_3d()
        move_dimension(cube, cube.coord('latitude'), -2)
        self.assertEqual(cube.shape, (2, 3, 4))

        cube = stock.simple_3d()
        move_dimension(cube, cube.coord('longitude'), 2)
        self.assertEqual(cube.shape, (2, 3, 4))

        cube = stock.simple_3d()
        move_dimension(cube, cube.coord('longitude'), -1)
        self.assertEqual(cube.shape, (2, 3, 4))

    def test_axis0_rolled_to_another_position(self):
        cube = stock.simple_3d()
        move_dimension(cube, 'wibble', 1)
        self.assertEqual(cube.shape, (3, 2, 4))

        cube = stock.simple_3d()
        move_dimension(cube, cube.coord('wibble'), 2)
        self.assertEqual(cube.shape, (3, 4, 2))

        cube = stock.simple_3d()
        move_dimension(cube, 'wibble', -1)
        self.assertEqual(cube.shape, (3, 4, 2))

        cube = stock.simple_3d()
        move_dimension(cube, 'wibble', -2)
        self.assertEqual(cube.shape, (3, 2, 4))

    def test_axis1_rolled_to_another_position(self):
        cube = stock.simple_3d()
        move_dimension(cube, 'latitude', 0)
        self.assertEqual(cube.shape, (3, 2, 4))

        cube = stock.simple_3d()
        move_dimension(cube, 'latitude', 2)
        self.assertEqual(cube.shape, (2, 4, 3))

        cube = stock.simple_3d()
        move_dimension(cube, 'latitude', -1)
        self.assertEqual(cube.shape, (2, 4, 3))

        cube = stock.simple_3d()
        move_dimension(cube, 'latitude', -3)
        self.assertEqual(cube.shape, (3, 2, 4))

    def test_axis2_rolled_to_another_position(self):
        cube = stock.simple_3d()
        move_dimension(cube, 'longitude', 0)
        self.assertEqual(cube.shape, (4, 2, 3))

        cube = stock.simple_3d()
        move_dimension(cube, 'longitude', 1)
        self.assertEqual(cube.shape, (2, 4, 3))

        cube = stock.simple_3d()
        move_dimension(cube, 'longitude', -2)
        self.assertEqual(cube.shape, (2, 4, 3))

        cube = stock.simple_3d()
        move_dimension(cube, 'longitude', -3)
        self.assertEqual(cube.shape, (4, 2, 3))

    def test_position_not_in_data(self):

        cube = stock.simple_3d()
        self.assertRaises(ValueError,
                          move_dimension, cube, 'wibble', 3)

        self.assertRaises(ValueError,
                          move_dimension, cube, 'wibble', -4)

    def test_roll_1d_coord_in_a_cube_with_some_multidim_coords(self):
        cube = stock.simple_3d_w_multidim_coords()
        move_dimension(cube, 'wibble', 2)
        self.assertEqual(cube.shape, (3, 4, 2))

    def test_attempting_to_roll_a_multidim_coord(self):
        cube = stock.simple_3d_w_multidim_coords()
        self.assertRaises(iris.exceptions.CoordinateMultiDimError,
                          move_dimension, cube, 'bar')

    def test_incorrect_argument_values(self):
        cube = stock.simple_3d()
        self.assertRaises(ValueError, move_dimension, cube,
                          'wibble', -4)
        self.assertRaises(ValueError, move_dimension, cube,
                          'wibble', 4)

    def test_name_or_coord_argument_is_wrong_type(self):
        cube = stock.simple_3d()
        self.assertRaises(TypeError, move_dimension, cube, 0)


if __name__ == '__main__':
    unittest.main()
