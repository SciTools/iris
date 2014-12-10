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
"""Unit tests for the `iris.fileformats.netcdf.Saver` class."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import netCDF4 as nc
import numpy as np

import iris
from iris.coord_systems import GeogCS, TransverseMercator, RotatedGeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf import Saver
import iris.tests.stock as stock


class Test_write(tests.IrisTest):
    def _transverse_mercator_cube(self, ellipsoid=None):
        data = np.arange(12).reshape(3, 4)
        cube = Cube(data, 'air_pressure_anomaly')
        trans_merc = TransverseMercator(49.0, -2.0, -400000.0, 100000.0,
                                        0.9996012717, ellipsoid)
        coord = DimCoord(range(3), 'projection_y_coordinate', units='m',
                         coord_system=trans_merc)
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(range(4), 'projection_x_coordinate', units='m',
                         coord_system=trans_merc)
        cube.add_dim_coord(coord, 1)
        return cube

    def test_transverse_mercator(self):
        # Create a Cube with a transverse Mercator coordinate system.
        ellipsoid = GeogCS(6377563.396, 6356256.909)
        cube = self._transverse_mercator_cube(ellipsoid)
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def test_transverse_mercator_no_ellipsoid(self):
        # Create a Cube with a transverse Mercator coordinate system.
        cube = self._transverse_mercator_cube()
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path)

    def _simple_cube(self, dtype):
        data = np.arange(12, dtype=dtype).reshape(3, 4)
        points = np.arange(3, dtype=dtype)
        bounds = np.arange(6, dtype=dtype).reshape(3, 2)
        cube = Cube(data, 'air_pressure_anomaly')
        coord = DimCoord(points, bounds=bounds)
        cube.add_dim_coord(coord, 0)
        return cube

    def test_little_endian(self):
        # Create a Cube with little-endian data.
        cube = self._simple_cube('<f4')
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path, basename='endian', flags='')

    def test_big_endian(self):
        # Create a Cube with big-endian data.
        cube = self._simple_cube('>f4')
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path, basename='endian', flags='')

    def test_zlib(self):
        cube = self._simple_cube('>f4')
        with mock.patch('iris.fileformats.netcdf.netCDF4') as api:
            with Saver('/dummy/path', 'NETCDF4') as saver:
                saver.write(cube, zlib=True)
        dataset = api.Dataset.return_value
        create_var_calls = mock.call.createVariable(
            'air_pressure_anomaly', np.dtype('float32'), ['dim0', 'dim1'],
            shuffle=True, least_significant_digit=None,
            contiguous=False, zlib=True, fletcher32=False,
            endian='native', complevel=4, chunksizes=None).call_list()
        dataset.assert_has_calls(create_var_calls)

    def test_least_significant_digit(self):
        cube = Cube(np.array([1.23, 4.56, 7.89]),
                    standard_name='surface_temperature', long_name=None,
                    var_name='temp', units='K')
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube, least_significant_digit=1)
            cube_saved = iris.load_cube(nc_path)
            self.assertEquals(
                cube_saved.attributes['least_significant_digit'], 1)
            self.assertFalse(np.all(cube.data == cube_saved.data))
            self.assertArrayAllClose(cube.data, cube_saved.data, 0.1)

    def test_default_unlimited_dimensions(self):
        cube = self._simple_cube('>f4')
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            ds = nc.Dataset(nc_path)
            self.assertTrue(ds.dimensions['dim0'].isunlimited())
            self.assertFalse(ds.dimensions['dim1'].isunlimited())
            ds.close()

    def test_no_unlimited_dimensions(self):
        cube = self._simple_cube('>f4')
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube, unlimited_dimensions=[])
            ds = nc.Dataset(nc_path)
            for dim in ds.dimensions.itervalues():
                self.assertFalse(dim.isunlimited())
            ds.close()

    def test_invalid_unlimited_dimensions(self):
        cube = self._simple_cube('>f4')
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                # should not raise an exception
                saver.write(cube, unlimited_dimensions=['not_found'])

    def test_custom_unlimited_dimensions(self):
        cube = self._transverse_mercator_cube()
        unlimited_dimensions = ['projection_y_coordinate',
                                'projection_x_coordinate']
        # test coordinates by name
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube, unlimited_dimensions=unlimited_dimensions)
            ds = nc.Dataset(nc_path)
            for dim in unlimited_dimensions:
                self.assertTrue(ds.dimensions[dim].isunlimited())
            ds.close()
        # test coordinate arguments
        with self.temp_filename('.nc') as nc_path:
            coords = [cube.coord(dim) for dim in unlimited_dimensions]
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube, unlimited_dimensions=coords)
            ds = nc.Dataset(nc_path)
            for dim in unlimited_dimensions:
                self.assertTrue(ds.dimensions[dim].isunlimited())
            ds.close()

    def test_reserved_attributes(self):
        cube = self._simple_cube('>f4')
        cube.attributes['dimensions'] = 'something something_else'
        with self.temp_filename('.nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            ds = nc.Dataset(nc_path)
            res = ds.getncattr('dimensions')
            ds.close()
            self.assertEqual(res, 'something something_else')


class TestCoordSystems(tests.IrisTest):
    def cube_with_cs(self, coord_system,
                     names=['grid_longitude', 'grid_latitude']):
        cube = stock.lat_lon_cube()
        x, y = cube.coord('longitude'), cube.coord('latitude')
        x.coord_system = y.coord_system = coord_system
        for coord, name in zip([x, y], names):
            coord.rename(name)
        return cube

    def construct_cf_grid_mapping_variable(self, cube):
        # Calls the actual NetCDF saver with appropriate mocking, returning
        # the grid variable that gets created.
        grid_variable = mock.Mock(name='NetCDFVariable')
        create_var_fn = mock.Mock(side_effect=[grid_variable])
        dataset = mock.Mock(variables=[],
                            createVariable=create_var_fn)
        saver = mock.Mock(spec=Saver, _coord_systems=[],
                          _dataset=dataset)
        variable = mock.Mock()

        Saver._create_cf_grid_mapping(saver, cube, variable)
        self.assertEqual(create_var_fn.call_count, 1)
        self.assertEqual(variable.grid_mapping,
                         grid_variable.grid_mapping_name)
        return grid_variable

    def variable_attributes(self, mocked_variable):
        """Get the attributes dictionary from a mocked NetCDF variable."""
        # Get the attributes defined on the mock object.
        attributes = filter(lambda name: not name.startswith('_'),
                            sorted(mocked_variable.__dict__.keys()))
        attributes.remove('method_calls')
        return {key: getattr(mocked_variable, key) for key in attributes}

    def test_rotated_geog_cs(self):
        coord_system = RotatedGeogCS(37.5, 177.5, ellipsoid=GeogCS(6371229.0))
        cube = self.cube_with_cs(coord_system)
        expected = {'grid_mapping_name': 'rotated_latitude_longitude',
                    'north_pole_grid_longitude': 0.0,
                    'grid_north_pole_longitude': 177.5,
                    'grid_north_pole_latitude': 37.5,
                    'longitude_of_prime_meridian': 0.0,
                    'earth_radius': 6371229.0,
                    }

        grid_variable = self.construct_cf_grid_mapping_variable(cube)
        actual = self.variable_attributes(grid_variable)

        # To see obvious differences, check that they keys are the same.
        self.assertEqual(sorted(actual.keys()), sorted(expected.keys()))
        # Now check that the values are equivalent.
        self.assertEqual(actual, expected)

    def test_spherical_geog_cs(self):
        coord_system = GeogCS(6371229.0)
        cube = self.cube_with_cs(coord_system)
        expected = {'grid_mapping_name': 'latitude_longitude',
                    'longitude_of_prime_meridian': 0.0,
                    'earth_radius': 6371229.0
                    }

        grid_variable = self.construct_cf_grid_mapping_variable(cube)
        actual = self.variable_attributes(grid_variable)

        # To see obvious differences, check that they keys are the same.
        self.assertEqual(sorted(actual.keys()), sorted(expected.keys()))
        # Now check that the values are equivalent.
        self.assertEqual(actual, expected)

    def test_elliptic_geog_cs(self):
        coord_system = GeogCS(637, 600)
        cube = self.cube_with_cs(coord_system)
        expected = {'grid_mapping_name': 'latitude_longitude',
                    'longitude_of_prime_meridian': 0.0,
                    'semi_minor_axis': 600.0,
                    'semi_major_axis': 637.0,
                    }

        grid_variable = self.construct_cf_grid_mapping_variable(cube)
        actual = self.variable_attributes(grid_variable)

        # To see obvious differences, check that they keys are the same.
        self.assertEqual(sorted(actual.keys()), sorted(expected.keys()))
        # Now check that the values are equivalent.
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    tests.main()
