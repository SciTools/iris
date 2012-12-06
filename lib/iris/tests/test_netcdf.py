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
Test CF-NetCDF file loading and saving.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os
import warnings

import netCDF4 as nc
import numpy as np

import iris
import iris.std_names
import iris.util
import stock

@iris.tests.skip_data
class TestNetCDFLoad(tests.IrisTest):
    def test_load_global_xyt_total(self):
        # Test loading single xyt CF-netCDF file.
        cube = iris.load_cube(tests.get_data_path(('NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_global_xyt_total.cml')) 

    def test_load_global_xyt_hires(self):
        # Test loading another single xyt CF-netCDF file.
        cube = iris.load_cube(tests.get_data_path(('NetCDF', 'global', 'xyt', 'SMALL_hires_wind_u_for_ipcc4.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_global_xyt_hires.cml'))

    def test_load_global_xyzt_gems(self):
        # Test loading single xyzt CF-netCDF file (multi-cube).
        cubes = iris.load(tests.get_data_path(('NetCDF', 'global', 'xyz_t', 'GEMS_CO2_Apr2006.nc')))
        self.assertCML(cubes, ('netcdf', 'netcdf_global_xyzt_gems.cml'))
        
        # Check the masked array fill value is propogated through the data manager loading.
	lnsp = cubes[1]
        self.assertTrue(np.ma.isMaskedArray(lnsp.data))
        self.assertEqual(-32767.0, lnsp.data.fill_value)

    def test_load_global_xyzt_gems_iter(self):
        # Test loading stepped single xyzt CF-netCDF file (multi-cube).
        for i, cube in enumerate(iris.load(tests.get_data_path(('NetCDF', 'global', 'xyz_t', 'GEMS_CO2_Apr2006.nc')))):
            self.assertCML(cube, ('netcdf', 'netcdf_global_xyzt_gems_iter_%d.cml' % i))

    def test_load_rotated_xy_land(self):
        # Test loading single xy rotated pole CF-netCDF file.
        cube = iris.load_cube(tests.get_data_path(('NetCDF', 'rotated', 'xy', 'rotPole_landAreaFraction.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_rotated_xy_land.cml'))

        # Make sure the AuxCoords have lazy data.
        self.assertIsInstance(cube.coord('latitude')._points,
                              iris.aux_factory.LazyArray)

    def test_load_rotated_xyt_precipitation(self):
        # Test loading single xyt rotated pole CF-netCDF file.
        cube = iris.load_cube(tests.get_data_path(('NetCDF', 'rotated', 'xyt', 'new_rotPole_precipitation.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_rotated_xyt_precipitation.cml'))

    def test_cell_methods(self):
        # Test exercising CF-netCDF cell method parsing.
        cubes = iris.load(tests.get_data_path(('NetCDF', 'testing', 'cell_methods.nc')))

        # TEST_COMPAT mod - new cube merge doesn't sort in the same way - test can pass by manual sorting...
        cubes = iris.cube.CubeList(sorted(cubes, key=lambda cube:cube.name()))

        self.assertCML(cubes, ('netcdf', 'netcdf_cell_methods.cml'))

    def test_deferred_loading(self):
        # Test exercising CF-netCDF deferred loading and deferred slicing.
        # shape (31, 161, 320)
        cube = iris.load_cube(tests.get_data_path(('NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc')))
        
        # Consecutive index on same dimension.
        self.assertCML(cube[0], ('netcdf', 'netcdf_deferred_index_0.cml'))
        self.assertCML(cube[0][0], ('netcdf', 'netcdf_deferred_index_1.cml'))
        self.assertCML(cube[0][0][0], ('netcdf', 'netcdf_deferred_index_2.cml'))
        
        # Consecutive slice on same dimension.
        self.assertCML(cube[0:20], ('netcdf', 'netcdf_deferred_slice_0.cml'))
        self.assertCML(cube[0:20][0:10], ('netcdf', 'netcdf_deferred_slice_1.cml'))
        self.assertCML(cube[0:20][0:10][0:5], ('netcdf', 'netcdf_deferred_slice_2.cml'))
        
        # Consecutive tuple index on same dimension.
        self.assertCML(cube[(0, 8, 4, 2, 14, 12), ], ('netcdf', 'netcdf_deferred_tuple_0.cml'))
        self.assertCML(cube[(0, 8, 4, 2, 14, 12), ][(0, 2, 4, 1), ], ('netcdf', 'netcdf_deferred_tuple_1.cml'))
        subcube = cube[(0, 8, 4, 2, 14, 12), ][(0, 2, 4, 1), ][(1, 3), ]
        self.assertCML(subcube, ('netcdf', 'netcdf_deferred_tuple_2.cml'))
        
        # Consecutive mixture on same dimension.
        self.assertCML(cube[0:20:2][(9, 5, 8, 0), ][3], ('netcdf', 'netcdf_deferred_mix_0.cml'))
        self.assertCML(cube[(2, 7, 3, 4, 5, 0, 9, 10), ][2:6][3], ('netcdf', 'netcdf_deferred_mix_0.cml'))
        self.assertCML(cube[0][(0, 2), (1, 3)], ('netcdf', 'netcdf_deferred_mix_1.cml'))

    def test_units(self):
        # Test exercising graceful cube and coordinate units loading.
        cube0, cube1 = iris.load(tests.get_data_path(('NetCDF', 'testing', 'units.nc')))

        self.assertCML(cube0, ('netcdf', 'netcdf_units_0.cml'))
        self.assertCML(cube1, ('netcdf', 'netcdf_units_1.cml'))


class TestSave(tests.IrisTest):
    def test_hybrid(self):
        cube = stock.realistic_4d()

        # Write Cube to netCDF file.
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cube, file_out, netcdf_format='NETCDF3_CLASSIC')

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_realistic_4d.cdl'))
        os.remove(file_out)

    def test_no_hybrid(self):
        cube = stock.realistic_4d()
        cube.remove_aux_factory(cube.aux_factories[0])

        # Write Cube to netCDF file.
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cube, file_out, netcdf_format='NETCDF3_CLASSIC')

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_realistic_4d_no_hybrid.cdl'))
        os.remove(file_out)

    def test_scalar_cube(self):
        cube = stock.realistic_4d()[0, 0, 0, 0]

        with self.temp_filename(suffix='.nc') as filename:
            iris.save(cube, filename, netcdf_format='NETCDF3_CLASSIC')
            self.assertCDL(filename, ('netcdf', 'netcdf_save_realistic_0d.cdl'))


@iris.tests.skip_data
class TestNetCDFSave(tests.IrisTest):
    def setUp(self):
        pass

    def test_netcdf_save_format(self):
        # Read netCDF input file.
        file_in = tests.get_data_path(('NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc'))
        cube = iris.load_cube(file_in)

        file_out = iris.util.create_temp_filename(suffix='.nc')

        # Test default NETCDF4 file format saving.
        iris.save(cube, file_out)
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF4', 'Failed to save as NETCDF4 format')
        ds.close()

        # Test NETCDF4_CLASSIC file format saving.
        iris.save(cube, file_out, netcdf_format='NETCDF4_CLASSIC')
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF4_CLASSIC', 'Failed to save as NETCDF4_CLASSIC format')	
        ds.close()

        # Test NETCDF3_CLASSIC file format saving.
        iris.save(cube, file_out, netcdf_format='NETCDF3_CLASSIC')
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF3_CLASSIC', 'Failed to save as NETCDF3_CLASSIC format')
        ds.close()

        # Test NETCDF4_64BIT file format saving.
        iris.save(cube, file_out, netcdf_format='NETCDF3_64BIT')
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF3_64BIT', 'Failed to save as NETCDF3_64BIT format')
        ds.close()

        # Test invalid file format saving.
        with self.assertRaises(ValueError):
            iris.save(cube, file_out, netcdf_format='WIBBLE')

        os.remove(file_out)

    def test_netcdf_save_single(self):
        # Test saving a single CF-netCDF file.
        # Read PP input file. 
        file_in = tests.get_data_path(('PP', 'cf_processing', '000003000000.03.236.000128.1990.12.01.00.00.b.pp'))
        cube = iris.load_cube(file_in)
        
        # Write Cube to netCDF file.
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cube, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_single.cdl'))
        os.remove(file_out)

    # TODO investigate why merge now make time an AuxCoord rather than a DimCoord and why forecast_period is 'preferred'.
    def test_netcdf_save_multi(self):
        # Test saving multiple CF-netCDF files.
        # Read PP input file.
        file_in = tests.get_data_path(('PP', 'cf_processing', 'abcza_pa19591997_daily_29.b.pp'))
        cubes = iris.load(file_in)
        
        # Cannot currently pass multiple cubes to the saver.
        self.assertRaises(ValueError, iris.save, cubes, iris.util.create_temp_filename(suffix='.nc'))

        # Save multiple cubes to multiple files.
        for index, cube in enumerate(cubes):
            # Write Cube to netCDF file.
            file_out = iris.util.create_temp_filename(suffix='.nc')
            
            iris.save(cube, file_out)
            
            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ('netcdf', 'netcdf_save_multi_%d.cdl' % index))
            os.remove(file_out)

    def test_netcdf_hybrid_height(self):
        # Test saving a CF-netCDF file which contains an atmosphere hybrid height (dimensionless vertical) coordinate.
        # Read PP input file.
        file_in = tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog.pp'))
        cube = iris.load_cube(file_in, 'air_potential_temperature')

        # Write Cube to netCDF file.
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cube, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_hybrid_height.cdl'))

        # Read netCDF file.
        cube = iris.load_cube(file_out)

        # Check the PP read, netCDF write, netCDF read mechanism.
        self.assertCML(cube, ('netcdf', 'netcdf_save_load_hybrid_height.cml'))

        os.remove(file_out)

    def test_netcdf_save_ndim_auxiliary(self):
        # Test saving a CF-netCDF file with multi-dimensional auxiliary coordinates.
        # Read netCDF input file.
        file_in = tests.get_data_path(('NetCDF', 'rotated', 'xyt', 'new_rotPole_precipitation.nc'))
        cube = iris.load_cube(file_in)
        
        # Write Cube to nerCDF file.
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cube, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_ndim_auxiliary.cdl'))

        # Read the netCDF file.
        cube = iris.load_cube(file_out)

        # Check the netCDF read, write, read mechanism.
        self.assertCML(cube, ('netcdf', 'netcdf_save_load_ndim_auxiliary.cml'))

        os.remove(file_out)
        
    def test_trajectory(self):
        cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        
        # extract a trajectory
        x = cube.coord('longitude').points[:10]
        y = cube.coord('latitude').points[:10]
        sample_points = [('latitude', x), ('longitude', y)]
        traj = iris.analysis.trajectory.interpolate(cube, sample_points)
        
        # save, reload and check
        with self.temp_filename(suffix='.nc') as temp_filename:
            iris.save(traj, temp_filename)
            reloaded = iris.load_cube(temp_filename)
            self.assertCML(reloaded, ('netcdf', 'save_load_traj.cml'))


class TestCFStandardName(tests.IrisTest):
    def setUp(self):
        pass

    def test_std_name_lookup_pass(self):
        # Test performing a CF standard name look-up hit.
        self.assertTrue('time' in iris.std_names.STD_NAMES)
 
    def test_std_name_lookup_fail(self):
        # Test performing a CF standard name look-up miss.
        self.assertFalse('phenomenon_time' in iris.std_names.STD_NAMES)


@iris.tests.skip_data
class TestNetCDFUKmoProcessFlags(tests.IrisTest):
    def test_process_flags(self):
        # Test single process flags
        for _, process_desc in iris.fileformats.pp.LBPROC_PAIRS[1:]:
            # Get basic cube and set process flag manually
            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = (process_desc,)
    
            # Save cube to netCDF
            temp_filename = iris.util.create_temp_filename(".nc")
            iris.save(ll_cube, temp_filename)

            # Reload cube     
            cube = iris.load_cube(temp_filename)
            
            
            # Check correct number and type of flags
            self.assertTrue(len(cube.attributes["ukmo__process_flags"]) == 1, "Mismatch in number of process flags.")
            process_flag = cube.attributes["ukmo__process_flags"][0]
            self.assertEquals(process_flag, process_desc)
            
            os.remove(temp_filename)

        # Test mutiple process flags
        multiple_bit_values = ((128, 64), (4096, 1024), (8192, 1024))
        
        # Maps lbproc value to the process flags that should be created
        multiple_map = {bits : [iris.fileformats.pp.lbproc_map[bit] for bit in bits] for bits in multiple_bit_values}

        for bits, descriptions in multiple_map.iteritems():
        
            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = descriptions
            
            # Save cube to netCDF
            temp_filename = iris.util.create_temp_filename(".nc")
            iris.save(ll_cube, temp_filename)
            
            # Reload cube     
            cube = iris.load_cube(temp_filename)
            
            # Check correct number and type of flags
            process_flags = cube.attributes["ukmo__process_flags"]
            self.assertTrue(len(process_flags) == len(bits), "Mismatch in number of process flags.")
            self.assertEquals(set(process_flags), set(descriptions))

            os.remove(temp_filename)


if __name__ == "__main__":
    tests.main()
