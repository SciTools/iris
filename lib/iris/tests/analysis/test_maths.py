# (C) British Crown Copyright 2013 - 2014, Met Office
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


class Test_maths_broadcasting(tests.IrisTest):
    """
    Additional tests for iris.analysis.maths to test simple cube
    cube.data broadcasting in_add_subtract_common()

    """
    def setUp(self):
        self.cube1 = stock.simple_3d()
        iris.analysis.clear_phenomenon_identity(self.cube1)
        self.cube2 = stock.simple_3d_w_multidim_coords()
        iris.analysis.clear_phenomenon_identity(self.cube2)

    def test_same(self):
        # Addition/subtraction when both cubes are the same
        other = self.cube1
        new_cube = self.cube1 + other - other
        self.assertEqual(new_cube, self.cube1)

    def test_collapse_outer_dim_addsub(self):
        # Collapse other's outer dim (wibble: 0)
        other = self.cube1.collapsed('wibble', iris.analysis.MIN)
        new_cube = self.cube1 + other - other
        self.assertEqual(new_cube, self.cube1)

    def test_collapse_inner_dim(self):
        # Collapse an inner dim (latitude: 0)
        other = self.cube1.collapsed('latitude', iris.analysis.MIN)
        new_cube = self.cube1 + other - other
        self.assertEqual(new_cube, self.cube1)


if __name__ == "__main__":
    tests.main()
