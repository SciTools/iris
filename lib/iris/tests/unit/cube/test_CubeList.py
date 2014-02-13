# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for the `iris.cube.CubeList` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.cube import Cube, CubeList
from iris.coords import AuxCoord
import iris.exceptions


class Test_merge_cube(tests.IrisTest):
    def setUp(self):
        self.cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        self.cube1.add_aux_coord(AuxCoord([0], "height", units="m"))

    def differing_epochs_cubes(self):
        cubes = CubeList()
        reftimes = ['hours since 1970-01-01 00:00:00',
                    'hours since 1970-01-02 00:00:00']
        calendar = 'gregorian'
        for reftime in reftimes:
            units = iris.unit.Unit(reftime, calendar='gregorian')
            cube = self.cube1.copy()
            frt_aux = AuxCoord(0, standard_name='forecast_reference_time',
                               units=units)
            cube.add_aux_coord(frt_aux)
            cubes.append(cube)
        return cubes

    def test_pass(self):
        cube2 = self.cube1.copy()
        cube2.coord("height").points = [1]
        result = CubeList([self.cube1, cube2]).merge_cube()
        self.assertIsInstance(result, Cube)

    def test_fail(self):
        cube2 = self.cube1.copy()
        cube2.rename("not air temperature")
        with self.assertRaises(iris.exceptions.MergeError):
            CubeList([self.cube1, cube2]).merge_cube()

    def test_empty(self):
        with self.assertRaises(ValueError):
            CubeList([]).merge_cube()

    def test_single_cube(self):
        result = CubeList([self.cube1]).merge_cube()
        self.assertEqual(result, self.cube1)
        self.assertIsNot(result, self.cube1)

    def test_repeated_cube(self):
        with self.assertRaises(iris.exceptions.MergeError):
            CubeList([self.cube1, self.cube1]).merge_cube()

    def test_differing_epochs_cubes(self):
        result = self.differing_epochs_cubes().merge_cube()
        self.assertIsInstance(result, Cube)


if __name__ == "__main__":
    tests.main()
