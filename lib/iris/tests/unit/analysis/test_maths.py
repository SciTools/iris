# (C) British Crown Copyright 2010 - 2013, Met Office
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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import iris
import iris.analysis.maths
import iris.tests.stock


class TestBasicMaths(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.simple_1d()

    def test_type_error(self):
        with self.assertRaises(TypeError):
            iris.analysis.maths.exponentiate('not a cube', 2)
        with self.assertRaises(TypeError):
            iris.analysis.maths.multiply('not a cube', 2)
        with self.assertRaises(TypeError):
            iris.analysis.maths.multiply(self.cube, 'not a cube')
        with self.assertRaises(TypeError):
            iris.analysis.maths.multiply(self.cube, 'not a cube',
                                         in_place=True)
        # test addition separately because it currently follows a different
        # code path from general binary operators
        with self.assertRaises(TypeError):
            iris.analysis.maths.add('not a cube', 123)
