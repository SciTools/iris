# (C) British Crown Copyright 2010 - 2012, Met Office
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris. If not, see <http://www.gnu.org/licenses/>.

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy

import iris


class Test_add_dim_coord(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube(numpy.arange(4).reshape(2, 2))

    def test_no_dim(self):
        self.assertRaises(TypeError,
                          self.cube.add_dim_coord,
                          iris.coords.DimCoord(numpy.arange(2), "latitude"))
        
    def test_adding_aux_coord(self):
        coord = iris.coords.AuxCoord(numpy.arange(2), "latitude")
        with self.assertRaises(ValueError):
            self.cube.add_dim_coord(coord, 0)


if __name__ == "__main__":
    tests.main()