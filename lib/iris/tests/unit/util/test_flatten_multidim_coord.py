# (C) British Crown Copyright 2010 - 2019, Met Office
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
from iris.util import flatten_multidim_coord


class Test(tests.IrisTest):
    def test_coord_name_as_argument(self):
        cube_a = stock.simple_2d_w_multidim_coords()
        cube_b = flatten_multidim_coord(cube_a, 'bar')
        self.assertEqual(cube_b.shape, (12, ))

    def test_coord_instance_as_argument(self):
        cube_a = stock.simple_2d_w_multidim_coords()
        cube_b = flatten_multidim_coord(cube_a, cube_a.coord('bar').copy())
        self.assertEqual(cube_b.shape, (12, ))

    def test_flatten_2d_coord_on_3d_cube(self):
        cube_a = stock.simple_3d_w_multidim_coords()
        coord = cube_a.coord('bar').copy()
        cube_b = flatten_multidim_coord(cube_a, coord)
        self.assertEqual(cube_b.dim_coords, (cube_a.coord('wibble'), ))
        self.assertEqual(cube_b.shape, (2, 12))


if __name__ == '__main__':
    unittest.main()
