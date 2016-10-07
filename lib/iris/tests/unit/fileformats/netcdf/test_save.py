# (C) British Crown Copyright 2014 - 2016, Met Office
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
"""Unit tests for the `iris.fileformats.netcdf.save` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import netCDF4 as nc
import numpy as np

import iris
from iris.cube import Cube
from iris.fileformats.netcdf import save, CF_CONVENTIONS_VERSION
from iris.tests.stock import lat_lon_cube


class Test_attributes(tests.IrisTest):
    def test_custom_conventions(self):
        # Ensure that we drop existing conventions attributes and replace with
        # CF convention.
        cube = Cube([0])
        cube.attributes['Conventions'] = 'convention1 convention2'

        with self.temp_filename('.nc') as nc_path:
            save(cube, nc_path, 'NETCDF4')
            ds = nc.Dataset(nc_path)
            res = ds.getncattr('Conventions')
            ds.close()
        self.assertEqual(res, CF_CONVENTIONS_VERSION)

    def test_attributes_arrays(self):
        # Ensure that attributes containing NumPy arrays can be equality
        # checked and their cubes saved as appropriate.
        c1 = Cube([], attributes={'bar': np.arange(2)})
        c2 = Cube([], attributes={'bar': np.arange(2)})

        with self.temp_filename('foo.nc') as nc_out:
            save([c1, c2], nc_out)
            ds = nc.Dataset(nc_out)
            res = ds.getncattr('bar')
            ds.close()
        self.assertArrayEqual(res, np.arange(2))

    def test_no_special_attribute_clash(self):
        # Ensure that saving multiple cubes with netCDF4 protected attributes
        # works as expected.
        # Note that here we are testing variable attribute clashes only - by
        # saving multiple cubes the attributes are saved as variable
        # attributes rather than global attributes.
        c1 = Cube([0], var_name='test', attributes={'name': 'bar'})
        c2 = Cube([0], var_name='test_1', attributes={'name': 'bar_1'})

        with self.temp_filename('foo.nc') as nc_out:
            save([c1, c2], nc_out)
            ds = nc.Dataset(nc_out)
            res = ds.variables['test'].getncattr('name')
            res_1 = ds.variables['test_1'].getncattr('name')
            ds.close()
        self.assertEqual(res, 'bar')
        self.assertEqual(res_1, 'bar_1')


class Test_unlimited_dims(tests.IrisTest):
    def test_no_unlimited_default(self):
        cube = lat_lon_cube()
        with iris.FUTURE.context(netcdf_no_unlimited=False):
            with self.temp_filename('foo.nc') as nc_out:
                save(cube, nc_out)
                ds = nc.Dataset(nc_out)
                self.assertTrue(ds.dimensions['latitude'].isunlimited())

    def test_no_unlimited_future_default(self):
        cube = lat_lon_cube()
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            with self.temp_filename('foo.nc') as nc_out:
                save(cube, nc_out)
                ds = nc.Dataset(nc_out)
                self.assertFalse(ds.dimensions['latitude'].isunlimited())


@tests.skip_data
class Test_packed_data(tests.IrisTest):
    def _get_scale_factor_add_offset(self, cube, datatype):
        dt = np.dtype(datatype)
        cmax = cube.data.max()
        cmin = cube.data.min()
        n = dt.itemsize * 8
        if isinstance(cube.data, np.ma.core.MaskedArray):
            masked = True
        else:
            masked = False
        if masked:
            scale_factor = (cmax - cmin)/(2**n-2)
        else:
            scale_factor = (cmax - cmin)/(2**n-1)
        if dt.kind == 'u':
            add_offset = cmin
        elif dt.kind == 'i':
            if masked:
                add_offset = (cmax + cmin)/2
            else:
                add_offset = cmin + 2**(n-1)*scale_factor
        return (scale_factor, add_offset)

    def test_single_packed_signed(self):
        """Test saving a single CF-netCDF file with packing."""
        # Read PP input file.
        file_in = tests.get_data_path(
            ('PP', 'cf_processing',
             '000003000000.03.236.000128.1990.12.01.00.00.b.pp'))
        cube = iris.load_cube(file_in)
        datatype = 'i2'
        scale_factor, offset = self._get_scale_factor_add_offset(cube,
                                                                 datatype)
        # Write Cube to netCDF file.
        with self.temp_filename(suffix='.nc') as file_out:
            iris.save(cube, file_out, pack_dtype=datatype)
            decimal = int(-np.log10(scale_factor))
            packedcube = iris.load_cube(file_out)
            # Check that packed cube is accurate to expected precision
            self.assertArrayAlmostEqual(cube.data, packedcube.data,
                                        decimal=decimal)
            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ('unit', 'fileformats', 'netcdf',
                                      'save',
                                      'single_packed_signed.cdl'))

    def test_single_packed_unsigned(self):
        """Test saving a single CF-netCDF file with packing into unsigned. """
        # Read PP input file.
        file_in = tests.get_data_path(
            ('PP', 'cf_processing',
             '000003000000.03.236.000128.1990.12.01.00.00.b.pp'))
        cube = iris.load_cube(file_in)
        datatype = 'u1'
        scale_factor, offset = self._get_scale_factor_add_offset(cube,
                                                                 datatype)
        # Write Cube to netCDF file.
        with self.temp_filename(suffix='.nc') as file_out:
            iris.save(cube, file_out, pack_dtype=datatype)
            decimal = int(-np.log10(scale_factor))
            packedcube = iris.load_cube(file_out)
            # Check that packed cube is accurate to expected precision
            self.assertArrayAlmostEqual(cube.data, packedcube.data,
                                        decimal=decimal)
            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ('unit', 'fileformats', 'netcdf',
                                      'save',
                                      'single_packed_unsigned.cdl'))

    def test_single_packed_manual_scale(self):
        """Test saving a single CF-netCDF file with packing with scale
        factor and add_offset set manually."""
        file_in = tests.get_data_path(
            ('PP', 'cf_processing',
             '000003000000.03.236.000128.1990.12.01.00.00.b.pp'))
        cube = iris.load_cube(file_in)
        datatype = 'i2'
        scale_factor, offset = self._get_scale_factor_add_offset(cube,
                                                                 datatype)
        cube.attributes['scale_factor'] = scale_factor
        cube.attributes['add_offset'] = offset

        # Write Cube to netCDF file.
        with self.temp_filename(suffix='.nc') as file_out:
            from shutil import copyfile
            iris.save(cube, file_out, pack_dtype=datatype)
            decimal = int(-np.log10(scale_factor))
            packedcube = iris.load_cube(file_out)
            # Check that packed cube is accurate to expected precision
            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ('unit', 'fileformats', 'netcdf',
                                      'save',
                                      'single_packed_manual.cdl'))
            self.assertArrayAlmostEqual(cube.data, packedcube.data,
                                        decimal=decimal)

    def test_multi_packed_single_dtype(self):
        """Test saving multiple packed cubes with the same pack_dtype."""
        # Read PP input file.
        file_in = tests.get_data_path(('PP', 'cf_processing',
                                       'abcza_pa19591997_daily_29.b.pp'))
        cubes = iris.load(file_in)
        dtype = 'i2'
        # Write Cube to netCDF file.
        with self.temp_filename(suffix='.nc') as file_out:
            iris.save(cubes, file_out, pack_dtype=dtype)
            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ('unit', 'fileformats', 'netcdf',
                                      'save',
                                      'multi_packed_single_dtype.cdl'))
            packedcubes = iris.load(file_out)
            # ensure cube order is the same:
            cubes.sort(key=lambda cube: cube.cell_methods[0].method)
            packedcubes.sort(key=lambda cube: cube.cell_methods[0].method)
            for cube, packedcube in zip(cubes, packedcubes):
                sf, ao = self._get_scale_factor_add_offset(cube, dtype)
                decimal = int(-np.log10(sf))
                # Check that packed cube is accurate to expected precision
                self.assertArrayAlmostEqual(cube.data, packedcube.data,
                                            decimal=decimal)

    def test_multi_packed_multi_dtype(self):
        """Test saving multiple packed cubes with pack_dtype list."""
        # Read PP input file.
        file_in = tests.get_data_path(('PP', 'cf_processing',
                                       'abcza_pa19591997_daily_29.b.pp'))
        cubes = iris.load(file_in)
        dtypes = ['i2', None, 'u2']
        # Write Cube to netCDF file.
        with self.temp_filename(suffix='.nc') as file_out:
            iris.save(cubes, file_out, pack_dtype=dtypes)
            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ('unit', 'fileformats', 'netcdf',
                                      'save',
                                      'multi_packed_multi_dtype.cdl'))
            packedcubes = iris.load(file_out)
            # ensure cube order is the same:
            cubes.sort(key=lambda cube: cube.cell_methods[0].method)
            packedcubes.sort(key=lambda cube: cube.cell_methods[0].method)
            for cube, packedcube, dtype in zip(cubes, packedcubes, dtypes):
                if dtype:
                    sf, ao = self._get_scale_factor_add_offset(cube, dtype)
                    decimal = int(-np.log10(sf))
                    # Check that packed cube is accurate to expected precision
                    self.assertArrayAlmostEqual(cube.data, packedcube.data,
                                                decimal=decimal)
                else:
                    self.assertArrayEqual(cube.data, packedcube.data)


if __name__ == "__main__":
    tests.main()
