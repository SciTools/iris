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
"""Test function :func:`iris.util.flatten_multidim_coord`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest

import iris.tests.stock as stock
from iris.coords import DimCoord, AuxCoord
from iris.util import flatten_multidim_coord, flatten_cube


class Test(tests.IrisTest):
    def test_argument_is_basestring(self):
        cube_a = stock.simple_2d_w_multidim_coords()

        cube_b = flatten_multidim_coord(cube_a, 'bar')
        self.assertEqual(cube_b.shape, (12, ))

    def test_argument_is_coord_instance(self):
        cube_a = stock.simple_3d_w_multidim_coords()
        coord = cube_a.coord('bar').copy()
        cube_b = flatten_multidim_coord(cube_a, coord)
        self.assertEqual(cube_b.dim_coords, (cube_a.coord('wibble'), ))
        self.assertEqual(cube_b.shape, (2, 12))

    def test_simple_cube_flattened(self):
        cube_a = stock.simple_2d()
        cube_b = flatten_cube(cube_a)
        self.assertEqual(cube_b.dim_coords, tuple())
        self.assertEqual(cube_b.shape, (12, ))

    def test_oned_dim_coord_flattened(self):
        cube_a = stock.simple_3d_w_multidim_coords()
        cube_a.add_dim_coord(DimCoord([0, 1, 2], var_name='blah'), (1,))
        cube_b = flatten_cube(cube_a, (1, 2))
        self.assertEqual(cube_b.dim_coords, (cube_a.coord('wibble'), ))
        self.assertEqual(cube_b.coord('blah', dim_coords=False).shape, (12, ))
        self.assertEqual(cube_b.shape, (2, 12))

    def test_multiple_oned_dim_coords_flattened(self):
        cube_a = stock.simple_3d_w_multidim_coords()
        cube_a.add_dim_coord(DimCoord([0, 1, 2], var_name='blah'), (1,))
        cube_a.add_dim_coord(DimCoord([0, 1, 2, 3], var_name='blah2'), (2,))
        cube_b = flatten_cube(cube_a, (1, 2))
        self.assertEqual(cube_b.dim_coords, (cube_a.coord('wibble'), ))
        self.assertEqual(cube_b.coord('blah', dim_coords=False).shape, (12, ))
        self.assertEqual(cube_b.coord('blah2', dim_coords=False).shape, (12,))
        self.assertEqual(cube_b.shape, (2, 12))

    def test_oned_aux_coord_flattened(self):
        cube_a = stock.simple_3d_w_multidim_coords()
        cube_a.add_aux_coord(AuxCoord([0, 1, 2], var_name='blah'), (1,))
        cube_b = flatten_cube(cube_a, (1, 2))
        self.assertEqual(cube_b.dim_coords, (cube_a.coord('wibble'), ))
        self.assertEqual(cube_b.coord('blah', dim_coords=False).shape, (12, ))
        self.assertEqual(cube_b.shape, (2, 12))

    def test_aux_coords_leading(self):
        cube_a = stock.simple_3d_w_multidim_coords()
        # Move the aux coord dims to the front of the cube
        cube_a.transpose((1, 2, 0))
        cube_b = flatten_cube(cube_a, (0, 1))
        self.assertEqual(cube_b.dim_coords, (cube_a.coord('wibble'), ))
        self.assertEqual(cube_b.shape, (12, 2))

    def test_split_aux_coord_dims(self):
        cube_a = stock.simple_3d_w_multidim_coords()
        cube_a.transpose((1, 0, 2))
        cube_b = flatten_cube(cube_a, (0, 2))
        self.assertEqual(cube_b.dim_coords, (cube_a.coord('wibble'), ))
        self.assertEqual(cube_b.shape, (12, 2))


if __name__ == '__main__':
    unittest.main()
