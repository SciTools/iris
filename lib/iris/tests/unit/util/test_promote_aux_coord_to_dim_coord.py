# (C) British Crown Copyright 2010 - 2015, Met Office
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
"""Test function :func:`iris.util.promote_aux_coord_to_dim_coord`."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest

import iris
import iris.tests.stock as stock
from iris.util import promote_aux_coord_to_dim_coord


class Test(tests.IrisTest):
    def test_dimension_already_has_dimcoord(self):
        cube_a = stock.hybrid_height()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, 'model_level_number')
        self.assertEqual(cube_b.dim_coords,
                         (cube_a.coord('model_level_number'),))

    def test_old_dim_coord_is_now_aux_coord(self):
        cube_a = stock.hybrid_height()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, 'model_level_number')
        self.assertTrue(cube_a.coord('level_height') in cube_b.aux_coords)

    def test_argument_is_coord_instance(self):
        cube_a = stock.realistic_4d()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, cube_b.coord('level_height'))
        self.assertEqual(cube_b.dim_coords,
                         (cube_a.coord('time'), cube_a.coord('level_height'),
                          cube_a.coord('grid_latitude'),
                          cube_a.coord('grid_longitude')))

    def test_dimension_is_anonymous(self):
        cube_a = stock.realistic_4d()
        cube_b = cube_a.copy()
        cube_b.remove_coord('model_level_number')
        promote_aux_coord_to_dim_coord(cube_b, 'level_height')
        self.assertEqual(cube_b.dim_coords,
                         (cube_a.coord('time'), cube_a.coord('level_height'),
                          cube_a.coord('grid_latitude'),
                          cube_a.coord('grid_longitude')))

    def test_already_a_dim_coord(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        cube_b = cube_a.copy()
        promote_aux_coord_to_dim_coord(cube_b, 'dim1')
        self.assertEqual(cube_a, cube_b)

    def test_coord_of_that_name_does_not_exist(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            promote_aux_coord_to_dim_coord(cube_a, 'wibble')

    def test_coord_does_not_exist(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        coord = cube_a.coord('dim1').copy()
        coord.rename('new')
        with self.assertRaises(ValueError):
            promote_aux_coord_to_dim_coord(cube_a, coord)

    def test_argument_is_wrong_type(self):
        cube_a = stock.simple_1d()
        with self.assertRaises(TypeError):
            promote_aux_coord_to_dim_coord(cube_a, 0.0)

    def test_trying_to_promote_a_multidim_coord(self):
        cube_a = stock.simple_2d_w_multidim_coords()
        with self.assertRaises(ValueError):
            promote_aux_coord_to_dim_coord(cube_a, 'bar')

    def test_trying_to_promote_a_scalar_coord(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        with self.assertRaises(ValueError):
            promote_aux_coord_to_dim_coord(cube_a, 'an_other')

    def test_trying_to_promote_a_nonmonotonic_coord(self):
        cube_a = stock.hybrid_height()
        with self.assertRaises(ValueError):
            promote_aux_coord_to_dim_coord(cube_a, 'surface_altitude')


if __name__ == '__main__':
    unittest.main()
