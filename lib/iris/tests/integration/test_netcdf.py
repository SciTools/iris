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
"""Integration tests for loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris
from iris.cube import Cube, CubeList
import iris.tests.stock as stock


class TestHybridPressure(tests.IrisTest):
    def setUp(self):
        # Modify stock cube so it is suitable to have a
        # hybrid pressure factory added to it.
        cube = stock.realistic_4d_no_derived()
        cube.coord('surface_altitude').rename('surface_air_pressure')
        cube.coord('surface_air_pressure').units = 'Pa'
        cube.coord('level_height').rename('level_pressure')
        cube.coord('level_pressure').units = 'Pa'
        # Construct and add hybrid pressure factory.
        factory = iris.aux_factory.HybridPressureFactory(
            cube.coord('level_pressure'),
            cube.coord('sigma'),
            cube.coord('surface_air_pressure'))
        cube.add_aux_factory(factory)
        self.cube = cube

    def test_save(self):
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename)
            self.assertCDL(filename)

    def test_save_load_loop(self):
        # Tests an issue where the variable names in the formula
        # terms changed to the standard_names instead of the variable names
        # when loading a previously saved cube.
        with self.temp_filename(suffix='.nc') as filename, \
                self.temp_filename(suffix='.nc') as other_filename:
            iris.save(self.cube, filename)
            cube = iris.load_cube(filename)
            iris.save(cube, other_filename)
            other_cube = iris.load_cube(other_filename)
            self.assertEqual(cube, other_cube)


class TestSaveMultipleAuxFactories(tests.IrisTest):
    def test_hybrid_height_and_pressure(self):
        cube = stock.realistic_4d()
        cube.add_aux_coord(iris.coords.DimCoord(
            1200.0, long_name='level_pressure', units='hPa'))
        cube.add_aux_coord(iris.coords.DimCoord(
            0.5, long_name='other sigma'))
        cube.add_aux_coord(iris.coords.DimCoord(
            1000.0, long_name='surface_air_pressure', units='hPa'))
        factory = iris.aux_factory.HybridPressureFactory(
            cube.coord('level_pressure'),
            cube.coord('other sigma'),
            cube.coord('surface_air_pressure'))
        cube.add_aux_factory(factory)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(cube, filename)
            self.assertCDL(filename)

    def test_shared_primary(self):
        cube = stock.realistic_4d()
        factory = iris.aux_factory.HybridHeightFactory(
            cube.coord('level_height'),
            cube.coord('sigma'),
            cube.coord('surface_altitude'))
        factory.rename('another altitude')
        cube.add_aux_factory(factory)
        with self.temp_filename(suffix='.nc') as filename, \
                self.assertRaisesRegexp(ValueError, 'multiple aux factories'):
            iris.save(cube, filename)


class TestUmVersionAttribute(tests.IrisTest):
    def test_single_saves_as_global(self):
        cube = Cube([1.0], standard_name='air_temperature', units='K',
                    attributes={'um_version': '4.3'})
        with self.temp_filename('.nc') as nc_path:
            iris.save(cube, nc_path)
            self.assertCDL(nc_path)

    def test_multiple_same_saves_as_global(self):
        cube_a = Cube([1.0], standard_name='air_temperature', units='K',
                      attributes={'um_version': '4.3'})
        cube_b = Cube([1.0], standard_name='air_pressure', units='hPa',
                      attributes={'um_version': '4.3'})
        with self.temp_filename('.nc') as nc_path:
            iris.save(CubeList([cube_a, cube_b]), nc_path)
            self.assertCDL(nc_path)

    def test_multiple_different_saves_on_variables(self):
        cube_a = Cube([1.0], standard_name='air_temperature', units='K',
                      attributes={'um_version': '4.3'})
        cube_b = Cube([1.0], standard_name='air_pressure', units='hPa',
                      attributes={'um_version': '4.4'})
        with self.temp_filename('.nc') as nc_path:
            iris.save(CubeList([cube_a, cube_b]), nc_path)
            self.assertCDL(nc_path)


if __name__ == "__main__":
    tests.main()
