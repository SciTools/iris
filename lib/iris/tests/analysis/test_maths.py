# (C) British Crown Copyright 2013, Met Office
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
Test the iris.analysis.maths module.

"""
# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.analysis
import iris.tests.stock as stock


class Test_maths(tests.IrisTest):
    def test__add_subtract_common(self):
        cube1 = stock.simple_3d()
        iris.analysis.clear_phenomenon_identity(cube1)
        cube2 = stock.simple_3d_w_multidim_coords()
        iris.analysis.clear_phenomenon_identity(cube2)
        
        # Both cubes are the same
        other = cube1
        intrim_cube = cube1 + other
        new_cube = intrim_cube - other
        self.assertEqual(new_cube, cube1)
        # Collapse other's outer dim (wibble: 0)
        other = cube1.collapsed('wibble', iris.analysis.MIN)
        intrim_cube = cube1 + other
        new_cube = intrim_cube - other
        self.assertEqual(new_cube, cube1)        
        # Collapse an inner dim (latitude: 0)
        other = cube1.collapsed('latitude', iris.analysis.MIN)
        with self.assertRaises(ValueError):
            intrim_cube = cube1 + other
            new_cube = intrim_cube - other
        # 'Select' an inner dimension (latitude: 1)    
        other = cube1[:,0:1,:]
        intrim_cube = cube1 + other
        new_cube = intrim_cube - other
        self.assertEqual(new_cube, cube1)
        # Incompatible cubes (incompatible metadata)
        with self.assertRaises(TypeError):
            intrim_cube = cube1 + cube2
            new_cube = intrim_cube - cube2
        # Collapse outer dim from cube with auxiliary coords describing
        # multiple dimensions, currently fails due to NotImplemented
        other = cube2.collapsed('wibble', iris.analysis.MIN)
        with self.assertRaises(TypeError):
            intrim_cube = cube1 + cube2
            new_cube = intrim_cube - cube2
        # Only difference is phenomenon
        other = cube1.copy()
        iris.analysis.clear_phenomenon_identity(other)
        intrim_cube = cube1 + other
        new_cube = intrim_cube - other
        self.assertEqual(new_cube, cube1)


if __name__ == "__main__":
    tests.main()
