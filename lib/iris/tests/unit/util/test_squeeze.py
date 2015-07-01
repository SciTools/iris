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
"""Test function :func:`iris.util.squeeze`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest

import iris
import iris.tests.stock as stock


class Test(tests.IrisTest):

    def setUp(self):
        self.cube = stock.simple_2d_w_multidim_and_scalars()

    def test_no_change(self):
        self.assertEqual(self.cube, iris.util.squeeze(self.cube))

    def test_squeeze_one_dim(self):
        cube_3d = iris.util.new_axis(self.cube, scalar_coord='an_other')
        cube_2d = iris.util.squeeze(cube_3d)

        self.assertEqual(self.cube, cube_2d)

    def test_squeeze_two_dims(self):
        cube_3d = iris.util.new_axis(self.cube, scalar_coord='an_other')
        cube_4d = iris.util.new_axis(cube_3d, scalar_coord='air_temperature')

        self.assertEqual(self.cube, iris.util.squeeze(cube_4d))

    def test_squeeze_one_anonymous_dim(self):
        cube_3d = iris.util.new_axis(self.cube)
        cube_2d = iris.util.squeeze(cube_3d)

        self.assertEqual(self.cube, cube_2d)

    def test_squeeze_to_scalar_cube(self):
        cube_scalar = self.cube[0, 0]
        cube_1d = iris.util.new_axis(cube_scalar)

        self.assertEqual(cube_scalar, iris.util.squeeze(cube_1d))


if __name__ == '__main__':
    unittest.main()
