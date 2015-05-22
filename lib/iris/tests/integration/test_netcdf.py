# (C) British Crown Copyright 2014 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from contextlib import contextmanager
import mock
import numpy as np

import iris
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION
from iris.fileformats.netcdf import Saver
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

    def test_hybrid_height_cubes(self):
        hh1 = stock.simple_4d_with_hybrid_height()
        hh1.attributes['cube'] = 'hh1'
        hh2 = stock.simple_4d_with_hybrid_height()
        hh2.attributes['cube'] = 'hh2'
        sa = hh2.coord('surface_altitude')
        sa.points = sa.points * 10
        with self.temp_filename('.nc') as fname:
            iris.save([hh1, hh2], fname)
            cubes = iris.load(fname)
            cubes = sorted(cubes, key=lambda cube: cube.attributes['cube'])
            self.assertCML(cubes)

    def test_hybrid_height_cubes_on_dimension_coordinate(self):
        hh1 = stock.hybrid_height()
        hh2 = stock.hybrid_height()
        sa = hh2.coord('surface_altitude')
        sa.points = sa.points * 10
        emsg = 'Unable to create dimensonless vertical coordinate.'
        with self.temp_filename('.nc') as fname, \
                self.assertRaisesRegexp(ValueError, emsg):
            iris.save([hh1, hh2], fname)


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


@contextmanager
def _patch_site_configuration():
    def cf_patch_conventions(conventions):
        return ', '.join([conventions, 'convention1, convention2'])

    def update(config):
        config['cf_profile'] = mock.Mock(name='cf_profile')
        config['cf_patch'] = mock.Mock(name='cf_patch')
        config['cf_patch_conventions'] = cf_patch_conventions

    orig_site_config = iris.site_configuration.copy()
    update(iris.site_configuration)
    yield
    iris.site_configuration = orig_site_config


class TestConventionsAttributes(tests.IrisTest):
    def test_patching_conventions_attribute(self):
        # Ensure that user defined conventions are wiped and those which are
        # saved patched through site_config can be loaded without an exception
        # being raised.
        cube = Cube([1.0], standard_name='air_temperature', units='K',
                    attributes={'Conventions':
                                'some user defined conventions'})

        # Patch the site configuration dictionary.
        with _patch_site_configuration(), self.temp_filename('.nc') as nc_path:
            iris.save(cube, nc_path)
            res = iris.load_cube(nc_path)

        self.assertEqual(res.attributes['Conventions'],
                         '{}, {}, {}'.format(CF_CONVENTIONS_VERSION,
                                             'convention1', 'convention2'))


class TestLazySave(tests.IrisTest):
    def test_lazy_preserved_save(self):
        fpath = tests.get_data_path(('NetCDF', 'label_and_climate',
                                     'small_FC_167_mon_19601101.nc'))
        acube = iris.load_cube(fpath)
        self.assertTrue(acube.has_lazy_data())
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(acube)
        self.assertTrue(acube.has_lazy_data())

    def test_lazy_mask_preserve_fill_value(self):
        cube = iris.cube.Cube(np.ma.array([0, 1], mask=[False, True],
                                          fill_value=-1))
        with self.temp_filename(suffix='.nc') as filename, \
                self.temp_filename(suffix='.nc') as other_filename:
            iris.save(cube, filename, unlimited_dimensions=[])
            ncube = iris.load_cube(filename)
            # Lazy save of the masked cube
            iris.save(ncube, other_filename, unlimited_dimensions=[])
            self.assertCDL(other_filename)


if __name__ == "__main__":
    tests.main()
