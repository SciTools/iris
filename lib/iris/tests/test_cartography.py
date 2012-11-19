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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Tests elements of the cartography module.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy

import iris
import iris.analysis.cartography

class Test_get_xy_grids(tests.IrisTest):
    
    def test_1d(self):
        
        cube = iris.cube.Cube(numpy.arange(100).reshape(10, 10))
        cube.add_dim_coord(iris.coords.DimCoord(numpy.arange(10), "latitude"), 0)
        cube.add_dim_coord(iris.coords.DimCoord(numpy.arange(10), "longitude"), 1)        
        
        x, y = iris.analysis.cartography.get_xy_grids(cube)
        self.assertRepr((x, y), ("cartography", "get_xy_grids", "1d.txt"))
        
    def test_2d(self):
        
        cube = iris.cube.Cube(numpy.arange(100).reshape(10, 10))
        cube.add_aux_coord(iris.coords.AuxCoord(cube.data, "latitude"), (0, 1))
        cube.add_aux_coord(iris.coords.AuxCoord(cube.data, "longitude"), (0, 1))        
        
        x, y = iris.analysis.cartography.get_xy_grids(cube)
        self.assertRepr((x, y), ("cartography", "get_xy_grids", "2d.txt"))
        
    def test_3d(self):
        
        cube = iris.cube.Cube(numpy.arange(1000).reshape(10, 10, 10))
        cube.add_aux_coord(iris.coords.AuxCoord(cube.data, "latitude"), (0, 1, 2))
        cube.add_aux_coord(iris.coords.AuxCoord(cube.data, "longitude"), (0, 1, 2))        

        self.assertRaises(ValueError, iris.analysis.cartography.get_xy_grids, cube)


if __name__ == "__main__":
    tests.main()
