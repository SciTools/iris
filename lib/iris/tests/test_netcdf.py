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
"""
Test CF-NetCDF file loading and saving.

"""

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import os
import shutil
import stat
import tempfile
import warnings

import mock
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

import iris
import iris.analysis.trajectory
import iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc as pyke_rules
import iris.fileformats.netcdf
import iris.std_names
import iris.util
import iris.coord_systems as icoord_systems
import iris.tests.stock as stock


def _write_nc_var(ds, name, dims=(), data=None, attributes={}):
    """Helper to create a new netCDF4 Variable in a dataset."""
    if data is None:
        datatype = 'c'
    else:
        data = np.array(data)
        datatype = data.dtype
    nc_var = ds.createVariable(name, datatype, dims)
    for att_name, att_val in attributes.iteritems():
        nc_var.setncattr(att_name, att_val)
    if data is not None:
        nc_var[:] = data


@iris.tests.skip_data
class TestNetCDFLoad(tests.IrisTest):
    def test_monotonic(self):
        cubes = iris.load(tests.get_data_path(
            ('NetCDF', 'testing', 'test_monotonic_coordinate.nc')))
        self.assertCML(cubes, ('netcdf', 'netcdf_monotonic.cml'))

    def test_load_global_xyt_total(self):
        # Test loading single xyt CF-netCDF file.
        cube = iris.load_cube(
            tests.get_data_path(('NetCDF', 'global', 'xyt',
                                 'SMALL_total_column_co2.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_global_xyt_total.cml'))

    def test_load_global_xyt_hires(self):
        # Test loading another single xyt CF-netCDF file.
        cube = iris.load_cube(tests.get_data_path(
            ('NetCDF', 'global', 'xyt', 'SMALL_hires_wind_u_for_ipcc4.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_global_xyt_hires.cml'))

    def test_missing_time_bounds(self):
        # Check we can cope with a missing bounds variable.
        with self.temp_filename(suffix='nc') as filename:
            # Tweak a copy of the test data file to rename (we can't delete)
            # the time bounds variable.
            src = tests.get_data_path(('NetCDF', 'global', 'xyt',
                                       'SMALL_hires_wind_u_for_ipcc4.nc'))
            shutil.copyfile(src, filename)
            dataset = nc.Dataset(filename, mode='a')
            dataset.renameVariable('time_bnds', 'foo')
            dataset.close()
            cube = iris.load_cube(filename, 'eastward_wind')

    def test_load_global_xyzt_gems(self):
        # Test loading single xyzt CF-netCDF file (multi-cube).
        cubes = iris.load(tests.get_data_path(('NetCDF', 'global', 'xyz_t',
                                               'GEMS_CO2_Apr2006.nc')))
        self.assertCML(cubes, ('netcdf', 'netcdf_global_xyzt_gems.cml'))

        # Check the masked array fill value is propogated through the data
        # manager loading.
        lnsp = cubes[1]
        self.assertTrue(ma.isMaskedArray(lnsp.data))
        self.assertEqual(-32767.0, lnsp.data.fill_value)

    def test_load_global_xyzt_gems_iter(self):
        # Test loading stepped single xyzt CF-netCDF file (multi-cube).
        for i, cube in enumerate(iris.load(
            tests.get_data_path(('NetCDF', 'global', 'xyz_t',
                                 'GEMS_CO2_Apr2006.nc')))):
            self.assertCML(cube, ('netcdf',
                                  'netcdf_global_xyzt_gems_iter_%d.cml' % i))

    def test_load_rotated_xy_land(self):
        # Test loading single xy rotated pole CF-netCDF file.
        cube = iris.load_cube(tests.get_data_path(
            ('NetCDF', 'rotated', 'xy', 'rotPole_landAreaFraction.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_rotated_xy_land.cml'))

        # Make sure the AuxCoords have lazy data.
        self.assertIsInstance(cube.coord('latitude')._points,
                              iris.aux_factory.LazyArray)

    def test_load_rotated_xyt_precipitation(self):
        # Test loading single xyt rotated pole CF-netCDF file.
        cube = iris.load_cube(
            tests.get_data_path(('NetCDF', 'rotated', 'xyt',
                                 'small_rotPole_precipitation.nc')))
        self.assertCML(cube, ('netcdf',
                              'netcdf_rotated_xyt_precipitation.cml'))

    def test_load_tmerc_grid_and_clim_bounds(self):
        # Test loading a single CF-netCDF file with a transverse Mercator
        # grid_mapping and a time variable with climatology.
        cube = iris.load_cube(
            tests.get_data_path(('NetCDF', 'transverse_mercator',
                                 'tmean_1910_1910.nc')))
        self.assertCML(cube, ('netcdf', 'netcdf_tmerc_and_climatology.cml'))

    def test_load_tmerc_grid_with_projection_origin(self):
        # Test loading a single CF-netCDF file with a transverse Mercator
        # grid_mapping that uses longitude_of_projection_origin and
        # scale_factor_at_projection_origin instead of
        # longitude_of_central_meridian and scale_factor_at_central_meridian.
        cube = iris.load_cube(
            tests.get_data_path(('NetCDF', 'transverse_mercator',
                                 'projection_origin_attributes.nc')))

        expected = icoord_systems.TransverseMercator(
            latitude_of_projection_origin=49.0,
            longitude_of_central_meridian=-2.0,
            false_easting=400000.0,
            false_northing=-100000.0,
            scale_factor_at_central_meridian=0.9996012717,
            ellipsoid=icoord_systems.GeogCS(
                semi_major_axis=6377563.396, semi_minor_axis=6356256.91))
        self.assertEqual(cube.coord('projection_x_coordinate').coord_system,
                         expected)
        self.assertEqual(cube.coord('projection_y_coordinate').coord_system,
                         expected)

    def _test_load_lambert_conformal(self, parallels, centre_latlons):
        # Check load of Lambert Conformal with given parameters.
        central_lat, central_lon = centre_latlons
        # create a temporary netcdf file + read a cube from it
        with self.temp_filename(suffix='.nc') as temp_ncpath:
            with nc.Dataset(temp_ncpath, 'w') as ds:
                nx, ny = 6, 4
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                _write_nc_var(ds, 'x', dims=('x'),
                              data=np.linspace(0.0, 100.0, nx),
                              attributes=
                                  {'units': 'km',
                                   'standard_name': 'projection_x_coordinate'})
                _write_nc_var(ds, 'y', dims=('y'),
                              data=np.linspace(0.0, 100.0, ny),
                              attributes=
                                  {'units': 'km',
                                   'standard_name': 'projection_y_coordinate'})
                _write_nc_var(
                    ds, 'Lambert_Conformal',
                    attributes={'grid_mapping_name':
                                'lambert_conformal_conic',
                                'standard_parallel': parallels,
                                'latitude_of_projection_origin': central_lat,
                                'longitude_of_central_meridian': central_lon})
                _write_nc_var(ds, 'temperature', dims=('y', 'x'),
                              data=np.zeros((ny, nx)),
                              attributes={'units': 'K',
                                          'standard_name': 'air_temperature',
                                          'grid_mapping': 'Lambert_Conformal'})
            # load file as a single cube
            cube, = iris.fileformats.netcdf.load_cubes([temp_ncpath])

        # test cube properties as required
        x_coord = cube.coord(axis='x')
        self.assertEqual(x_coord.name(), 'projection_x_coordinate')
        cs = x_coord.coord_system
        self.assertEqual(cs.grid_mapping_name, 'lambert_conformal')
        self.assertEqual(cs.central_lat, central_lat)
        self.assertEqual(cs.central_lon, central_lon)
        if len(parallels) == 1:
            self.assertEqual(cs.secant_latitudes,
                             (parallels[0], parallels[0]))
        else:
            self.assertEqual(cs.secant_latitudes, parallels)

    def test_load_lambert_conformal_1sp(self):
        # Check Lambert Conformal load with a single parallel.
        self._test_load_lambert_conformal(
            parallels=(50.0,),
            centre_latlons=(50.0, 107.0))

    def test_load_lambert_conformal_2sp(self):
        # Check Lambert Conformal load with two parallels.
        self._test_load_lambert_conformal(
            parallels=(-35.3, -73.0),
            centre_latlons=(-51.2, 137.0))

    def test_missing_climatology(self):
        # Check we can cope with a missing climatology variable.
        with self.temp_filename(suffix='nc') as filename:
            # Tweak a copy of the test data file to rename (we can't delete)
            # the climatology variable.
            src = tests.get_data_path(('NetCDF', 'transverse_mercator',
                                       'tmean_1910_1910.nc'))
            shutil.copyfile(src, filename)
            dataset = nc.Dataset(filename, mode='a')
            dataset.renameVariable('climatology_bounds', 'foo')
            dataset.close()
            cube = iris.load_cube(filename, 'Mean temperature')

    def test_cell_methods(self):
        # Test exercising CF-netCDF cell method parsing.
        cubes = iris.load(tests.get_data_path(('NetCDF', 'testing',
                                               'cell_methods.nc')))

        # TEST_COMPAT mod - new cube merge doesn't sort in the same way - test
        # can pass by manual sorting...
        cubes = iris.cube.CubeList(sorted(cubes, key=lambda cube: cube.name()))

        self.assertCML(cubes, ('netcdf', 'netcdf_cell_methods.cml'))

    def test_deferred_loading(self):
        # Test exercising CF-netCDF deferred loading and deferred slicing.
        # shape (31, 161, 320)
        cube = iris.load_cube(tests.get_data_path(
            ('NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc')))

        # Consecutive index on same dimension.
        self.assertCML(cube[0], ('netcdf', 'netcdf_deferred_index_0.cml'))
        self.assertCML(cube[0][0], ('netcdf', 'netcdf_deferred_index_1.cml'))
        self.assertCML(cube[0][0][0], ('netcdf',
                                       'netcdf_deferred_index_2.cml'))

        # Consecutive slice on same dimension.
        self.assertCML(cube[0:20], ('netcdf', 'netcdf_deferred_slice_0.cml'))
        self.assertCML(cube[0:20][0:10], ('netcdf',
                                          'netcdf_deferred_slice_1.cml'))
        self.assertCML(cube[0:20][0:10][0:5], ('netcdf',
                                               'netcdf_deferred_slice_2.cml'))

        # Consecutive tuple index on same dimension.
        self.assertCML(cube[(0, 8, 4, 2, 14, 12), ],
                       ('netcdf', 'netcdf_deferred_tuple_0.cml'))
        self.assertCML(cube[(0, 8, 4, 2, 14, 12), ][(0, 2, 4, 1), ],
                       ('netcdf', 'netcdf_deferred_tuple_1.cml'))
        subcube = cube[(0, 8, 4, 2, 14, 12), ][(0, 2, 4, 1), ][(1, 3), ]
        self.assertCML(subcube, ('netcdf', 'netcdf_deferred_tuple_2.cml'))

        # Consecutive mixture on same dimension.
        self.assertCML(cube[0:20:2][(9, 5, 8, 0), ][3],
                       ('netcdf', 'netcdf_deferred_mix_0.cml'))
        self.assertCML(cube[(2, 7, 3, 4, 5, 0, 9, 10), ][2:6][3],
                       ('netcdf', 'netcdf_deferred_mix_0.cml'))
        self.assertCML(cube[0][(0, 2), (1, 3)],
                       ('netcdf', 'netcdf_deferred_mix_1.cml'))

    def test_units(self):
        # Test exercising graceful cube and coordinate units loading.
        cube0, cube1 = iris.load(tests.get_data_path(('NetCDF', 'testing',
                                                      'units.nc')))

        self.assertCML(cube0, ('netcdf', 'netcdf_units_0.cml'))
        self.assertCML(cube1, ('netcdf', 'netcdf_units_1.cml'))


class TestNetCDFCRS(tests.IrisTest):
    def setUp(self):
        class Var(object):
            pass

        self.grid = Var()

    def test_lat_lon_major_minor(self):
        major = 63781370
        minor = 63567523
        self.grid.semi_major_axis = major
        self.grid.semi_minor_axis = minor
        crs = pyke_rules.build_coordinate_system(self.grid)
        self.assertEqual(crs, icoord_systems.GeogCS(major, minor))

    def test_lat_lon_earth_radius(self):
        earth_radius = 63700000
        self.grid.earth_radius = earth_radius
        crs = pyke_rules.build_coordinate_system(self.grid)
        self.assertEqual(crs, icoord_systems.GeogCS(earth_radius))


class SaverPermissions(tests.IrisTest):
    def test_noexist_directory(self):
        # Test capture of suitable exception raised on writing to a
        # non-existent directory.
        dir_name = os.path.join(tempfile.gettempdir(), 'non_existent_dir')
        fnme = os.path.join(dir_name, 'tmp.nc')
        with self.assertRaises(IOError):
            with iris.fileformats.netcdf.Saver(fnme, 'NETCDF4'):
                pass

    def test_bad_permissions(self):
        # Non-exhaustive check that wrong permissions results in a suitable
        # exception being raised.
        dir_name = tempfile.mkdtemp()
        fnme = os.path.join(dir_name, 'tmp.nc')
        try:
            os.chmod(dir_name, stat.S_IREAD)
            with self.assertRaises(IOError):
                iris.fileformats.netcdf.Saver(fnme, 'NETCDF4')
            self.assertFalse(os.path.exists(fnme))
        finally:
            os.rmdir(dir_name)


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
        self.assertCDL(file_out, ('netcdf',
                                  'netcdf_save_realistic_4d_no_hybrid.cdl'))
        os.remove(file_out)

    def test_scalar_cube(self):
        cube = stock.realistic_4d()[0, 0, 0, 0]

        with self.temp_filename(suffix='.nc') as filename:
            iris.save(cube, filename, netcdf_format='NETCDF3_CLASSIC')
            self.assertCDL(filename, ('netcdf',
                                      'netcdf_save_realistic_0d.cdl'))

    def test_no_name_cube(self):
        # Cube with no names.
        cube = iris.cube.Cube(np.arange(20, dtype=np.float64).reshape((4, 5)))
        dim0 = iris.coords.DimCoord(np.arange(4, dtype=np.float64))
        dim1 = iris.coords.DimCoord(np.arange(5, dtype=np.float64), units='m')
        other = iris.coords.AuxCoord('foobar', units='no_unit')
        cube.add_dim_coord(dim0, 0)
        cube.add_dim_coord(dim1, 1)
        cube.add_aux_coord(other)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(cube, filename, netcdf_format='NETCDF3_CLASSIC')
            self.assertCDL(filename, ('netcdf', 'netcdf_save_no_name.cdl'))


class TestNetCDFSave(tests.IrisTest):
    def setUp(self):
        self.cubell = iris.cube.Cube(np.arange(4).reshape(2, 2),
                                     'air_temperature')
        self.cube = iris.cube.Cube(np.zeros([2, 2]),
                                   standard_name='surface_temperature',
                                   long_name=None,
                                   var_name='temp',
                                   units='K')
        self.cube2 = iris.cube.Cube(np.ones([1, 2, 2]),
                                    standard_name=None,
                                    long_name='Something Random',
                                    var_name='temp2',
                                    units='K')
        self.cube3 = iris.cube.Cube(np.ones([2, 2, 2]),
                                    standard_name=None,
                                    long_name='Something Random',
                                    var_name='temp3',
                                    units='K')
        self.cube4 = iris.cube.Cube(np.zeros([10]),
                                    standard_name='air_temperature',
                                    long_name=None,
                                    var_name='temp',
                                    units='K')
        self.cube5 = iris.cube.Cube(np.ones([20]),
                                    standard_name=None,
                                    long_name='air_temperature',
                                    var_name='temp2',
                                    units='K')
        self.cube6 = iris.cube.Cube(np.ones([10]),
                                    standard_name=None,
                                    long_name='air_temperature',
                                    var_name='temp3',
                                    units='K')

    @iris.tests.skip_data
    def test_netcdf_save_format(self):
        # Read netCDF input file.
        file_in = tests.get_data_path(
            ('NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc'))
        cube = iris.load_cube(file_in)

        file_out = iris.util.create_temp_filename(suffix='.nc')

        # Test default NETCDF4 file format saving.
        iris.save(cube, file_out)
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF4',
                         'Failed to save as NETCDF4 format')
        ds.close()

        # Test NETCDF4_CLASSIC file format saving.
        iris.save(cube, file_out, netcdf_format='NETCDF4_CLASSIC')
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF4_CLASSIC',
                         'Failed to save as NETCDF4_CLASSIC format')
        ds.close()

        # Test NETCDF3_CLASSIC file format saving.
        iris.save(cube, file_out, netcdf_format='NETCDF3_CLASSIC')
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF3_CLASSIC',
                         'Failed to save as NETCDF3_CLASSIC format')
        ds.close()

        # Test NETCDF4_64BIT file format saving.
        iris.save(cube, file_out, netcdf_format='NETCDF3_64BIT')
        ds = nc.Dataset(file_out)
        self.assertEqual(ds.file_format, 'NETCDF3_64BIT',
                         'Failed to save as NETCDF3_64BIT format')
        ds.close()

        # Test invalid file format saving.
        with self.assertRaises(ValueError):
            iris.save(cube, file_out, netcdf_format='WIBBLE')

        os.remove(file_out)

    @iris.tests.skip_data
    def test_netcdf_save_single(self):
        # Test saving a single CF-netCDF file.
        # Read PP input file.
        file_in = tests.get_data_path(
            ('PP', 'cf_processing',
             '000003000000.03.236.000128.1990.12.01.00.00.b.pp'))
        cube = iris.load_cube(file_in)

        # Write Cube to netCDF file.
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cube, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_single.cdl'))
        os.remove(file_out)

    # TODO investigate why merge now make time an AuxCoord rather than a
    # DimCoord and why forecast_period is 'preferred'.
    @iris.tests.skip_data
    def test_netcdf_save_multi2multi(self):
        # Test saving multiple CF-netCDF files.
        # Read PP input file.
        file_in = tests.get_data_path(('PP', 'cf_processing',
                                       'abcza_pa19591997_daily_29.b.pp'))
        cubes = iris.load(file_in)

        # Save multiple cubes to multiple files.
        for index, cube in enumerate(cubes):
            # Write Cube to netCDF file.
            file_out = iris.util.create_temp_filename(suffix='.nc')

            iris.save(cube, file_out)

            # Check the netCDF file against CDL expected output.
            self.assertCDL(file_out, ('netcdf',
                                      'netcdf_save_multi_%d.cdl' % index))
            os.remove(file_out)

    @iris.tests.skip_data
    def test_netcdf_save_multi2single(self):
        # Test saving multiple cubes to a single CF-netCDF file.
        # Read PP input file.
        file_in = tests.get_data_path(('PP', 'cf_processing',
                                       'abcza_pa19591997_daily_29.b.pp'))
        cubes = iris.load(file_in)

        # Write Cube to netCDF file.
        file_out = iris.util.create_temp_filename(suffix='.nc')

        # Check that it is the same on loading
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_multiple.cdl'))

    def test_netcdf_multi_nocoord(self):
        # Testing the saving of a cublist with no coords.
        cubes = iris.cube.CubeList([self.cube, self.cube2, self.cube3])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_nocoord.cdl'))
        os.remove(file_out)

    def test_netcdf_multi_samevarnme(self):
        # Testing the saving of a cublist with cubes of the same var_name.
        self.cube2.var_name = self.cube.var_name
        cubes = iris.cube.CubeList([self.cube, self.cube2])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_samevar.cdl'))
        os.remove(file_out)

    def test_netcdf_multi_with_coords(self):
        # Testing the saving of a cublist with coordinates.
        lat = iris.coords.DimCoord(np.arange(2),
                                   long_name=None, var_name='lat',
                                   units='degree_north')
        lon = iris.coords.DimCoord(np.arange(2), standard_name='longitude',
                                   long_name=None, var_name='lon',
                                   units='degree_east')
        rcoord = iris.coords.DimCoord(np.arange(1), standard_name=None,
                                      long_name='Rnd Coordinate',
                                      units=None)
        self.cube.add_dim_coord(lon, 0)
        self.cube.add_dim_coord(lat, 1)
        self.cube2.add_dim_coord(lon, 1)
        self.cube2.add_dim_coord(lat, 2)
        self.cube2.add_dim_coord(rcoord, 0)

        cubes = iris.cube.CubeList([self.cube, self.cube2])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_wcoord.cdl'))
        os.remove(file_out)

    def test_netcdf_multi_wtih_samedimcoord(self):
        time1 = iris.coords.DimCoord(np.arange(10),
                                     standard_name='time',
                                     var_name='time')
        time2 = iris.coords.DimCoord(np.arange(20),
                                     standard_name='time',
                                     var_name='time')

        self.cube4.add_dim_coord(time1, 0)
        self.cube5.add_dim_coord(time2, 0)
        self.cube6.add_dim_coord(time1, 0)

        cubes = iris.cube.CubeList([self.cube4, self.cube5, self.cube6])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_samedimcoord.cdl'))
        os.remove(file_out)

    def test_netcdf_multi_conflict_name_dup_coord(self):
        # Duplicate coordinates with modified variable names lookup.
        latitude1 = iris.coords.DimCoord(np.arange(10),
                                         standard_name='latitude')
        time2 = iris.coords.DimCoord(np.arange(2),
                                     standard_name='time')
        latitude2 = iris.coords.DimCoord(np.arange(2),
                                         standard_name='latitude')

        self.cube6.add_dim_coord(latitude1, 0)
        self.cube.add_dim_coord(latitude2[:], 1)
        self.cube.add_dim_coord(time2[:], 0)

        cubes = iris.cube.CubeList([self.cube, self.cube6, self.cube6.copy()])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf',
                                  'multi_dim_coord_slightly_different.cdl'))
        os.remove(file_out)

    @iris.tests.skip_data
    def test_netcdf_hybrid_height(self):
        # Test saving a CF-netCDF file which contains a hybrid height
        # (i.e. dimensionless vertical) coordinate.
        # Read PP input file.
        file_in = tests.get_data_path(
            ('PP', 'COLPEX', 'small_colpex_theta_p_alt.pp'))
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

    @iris.tests.skip_data
    def test_netcdf_save_ndim_auxiliary(self):
        # Test saving CF-netCDF with multi-dimensional auxiliary coordinates.
        # Read netCDF input file.
        file_in = tests.get_data_path(
            ('NetCDF', 'rotated', 'xyt', 'small_rotPole_precipitation.nc'))
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

    def test_netcdf_save_conflicting_aux(self):
        # Test saving CF-netCDF with multi-dimensional auxiliary coordinates,
        # with conflicts.
        self.cube4.add_aux_coord(iris.coords.AuxCoord(np.arange(10),
                                                      'time'), 0)
        self.cube6.add_aux_coord(iris.coords.AuxCoord(np.arange(10, 20),
                                                      'time'), 0)

        cubes = iris.cube.CubeList([self.cube4, self.cube6])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_conf_aux.cdl'))
        os.remove(file_out)

    def test_netcdf_save_gridmapping(self):
        # Test saving CF-netCDF from a cubelist with various grid mappings.

        c1 = self.cubell
        c2 = self.cubell.copy()
        c3 = self.cubell.copy()

        coord_system = icoord_systems.GeogCS(6371229)
        coord_system2 = icoord_systems.GeogCS(6371228)
        coord_system3 = icoord_systems.RotatedGeogCS(30, 30)

        c1.add_dim_coord(iris.coords.DimCoord(
            np.arange(1, 3), 'latitude', long_name='1',
            coord_system=coord_system), 1)
        c1.add_dim_coord(iris.coords.DimCoord(
            np.arange(1, 3), 'longitude', long_name='1',
            coord_system=coord_system), 0)

        c2.add_dim_coord(iris.coords.DimCoord(
            np.arange(1, 3), 'latitude', long_name='2',
            coord_system=coord_system2), 1)
        c2.add_dim_coord(iris.coords.DimCoord(
            np.arange(1, 3), 'longitude', long_name='2',
            coord_system=coord_system2), 0)

        c3.add_dim_coord(iris.coords.DimCoord(
            np.arange(1, 3), 'grid_latitude', long_name='3',
            coord_system=coord_system3), 1)
        c3.add_dim_coord(iris.coords.DimCoord(
            np.arange(1, 3), 'grid_longitude', long_name='3',
            coord_system=coord_system3), 0)

        cubes = iris.cube.CubeList([c1, c2, c3])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_gridmapmulti.cdl'))
        os.remove(file_out)

    def test_netcdf_save_conflicting_names(self):
        # Test saving CF-netCDF with a dimension name corresponding to
        # an existing variable name (conflict).
        self.cube4.add_dim_coord(iris.coords.DimCoord(np.arange(10),
                                                      'time'), 0)
        self.cube6.add_aux_coord(iris.coords.AuxCoord(1, 'time'), None)

        cubes = iris.cube.CubeList([self.cube4, self.cube6])
        file_out = iris.util.create_temp_filename(suffix='.nc')
        iris.save(cubes, file_out)

        # Check the netCDF file against CDL expected output.
        self.assertCDL(file_out, ('netcdf', 'netcdf_save_conf_name.cdl'))
        os.remove(file_out)

    @iris.tests.skip_data
    def test_trajectory(self):
        file_in = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        cube = iris.load_cube(file_in)

        # extract a trajectory
        xpoint = cube.coord('longitude').points[:10]
        ypoint = cube.coord('latitude').points[:10]
        sample_points = [('latitude', xpoint), ('longitude', ypoint)]
        traj = iris.analysis.trajectory.interpolate(cube, sample_points)

        # save, reload and check
        with self.temp_filename(suffix='.nc') as temp_filename:
            iris.save(traj, temp_filename)
            reloaded = iris.load_cube(temp_filename)
            self.assertCML(reloaded, ('netcdf', 'save_load_traj.cml'))

    def test_attributes(self):
        # Should be global attributes.
        self.cube.attributes['history'] = 'A long time ago...'
        self.cube.attributes['title'] = 'Attribute test'
        self.cube.attributes['foo'] = 'bar'
        # Should be data varible attributes.
        self.cube.attributes['standard_error_multiplier'] = 23
        self.cube.attributes['flag_masks'] = 'a'
        self.cube.attributes['flag_meanings'] = 'b'
        self.cube.attributes['flag_values'] = 'c'
        self.cube.attributes['STASH'] = iris.fileformats.pp.STASH(1, 2, 3)
        # Should be overriden.
        self.cube.attributes['conventions'] = 'TEST'
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename)
            self.assertCDL(filename, ('netcdf', 'netcdf_save_attr.cdl'))

    def test_conflicting_attributes(self):
        # Should be data variable attributes.
        self.cube.attributes['foo'] = 'bar'
        self.cube2.attributes['foo'] = 'orange'
        with self.temp_filename(suffix='.nc') as filename:
            iris.save([self.cube, self.cube2], filename)
            self.assertCDL(filename, ('netcdf', 'netcdf_save_confl_attr.cdl'))

    def test_conflicting_global_attributes(self):
        # Should be data variable attributes, but raise a warning.
        attr_name = 'history'
        self.cube.attributes[attr_name] = 'Team A won.'
        self.cube2.attributes[attr_name] = 'Team B won.'
        expected_msg = '{attr_name!r} is being added as CF data variable ' \
                       'attribute, but {attr_name!r} should only be a CF ' \
                       'global attribute.'.format(attr_name=attr_name)
        with self.temp_filename(suffix='.nc') as filename:
            with mock.patch('warnings.warn') as warn:
                iris.save([self.cube, self.cube2], filename)
                warn.assert_called_with(expected_msg)
                self.assertCDL(filename,
                               ('netcdf', 'netcdf_save_confl_global_attr.cdl'))

    def test_no_global_attributes(self):
        # Should all be data variable attributes.
        # Different keys.
        self.cube.attributes['a'] = 'a'
        self.cube2.attributes['b'] = 'a'
        self.cube3.attributes['c'] = 'a'
        self.cube4.attributes['d'] = 'a'
        self.cube5.attributes['e'] = 'a'
        self.cube6.attributes['f'] = 'a'
        # Different values.
        self.cube.attributes['g'] = 'p'
        self.cube2.attributes['g'] = 'q'
        self.cube3.attributes['g'] = 'r'
        self.cube4.attributes['g'] = 's'
        self.cube5.attributes['g'] = 't'
        self.cube6.attributes['g'] = 'u'
        # One different value.
        self.cube.attributes['h'] = 'v'
        self.cube2.attributes['h'] = 'v'
        self.cube3.attributes['h'] = 'v'
        self.cube4.attributes['h'] = 'w'
        self.cube5.attributes['h'] = 'v'
        self.cube6.attributes['h'] = 'v'
        cubes = [self.cube, self.cube2, self.cube3,
                 self.cube4, self.cube5, self.cube6]
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(cubes, filename)
            self.assertCDL(filename, ('netcdf',
                                      'netcdf_save_no_global_attr.cdl'))


class TestNetCDF3SaveInteger(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube(np.zeros((2, 2), dtype=np.float64),
                                   standard_name='surface_temperature',
                                   long_name=None,
                                   var_name='temp',
                                   units='K')

    def test_int64_dimension_coord_netcdf3(self):
        coord = iris.coords.DimCoord(np.array([1, 2], dtype=np.int64),
                                     long_name='x')
        self.cube.add_dim_coord(coord, 0)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ('netcdf',
                                      'int64_dimension_coord_netcdf3.cml'),
                           checksum=False)

    def test_int64_auxiliary_coord_netcdf3(self):
        coord = iris.coords.AuxCoord(np.array([1, 2], dtype=np.int64),
                                     long_name='x')
        self.cube.add_aux_coord(coord, 0)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ('netcdf',
                                      'int64_auxiliary_coord_netcdf3.cml'),
                           checksum=False)

    def test_int64_data_netcdf3(self):
        self.cube.data = self.cube.data.astype(np.int64)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ('netcdf',
                                      'int64_data_netcdf3.cml'))

    def test_uint32_dimension_coord_netcdf3(self):
        coord = iris.coords.DimCoord(np.array([1, 2], dtype=np.uint32),
                                     long_name='x')
        self.cube.add_dim_coord(coord, 0)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ('netcdf',
                                      'uint32_dimension_coord_netcdf3.cml'),
                           checksum=False)

    def test_uint32_auxiliary_coord_netcdf3(self):
        coord = iris.coords.AuxCoord(np.array([1, 2], dtype=np.uint32),
                                     long_name='x')
        self.cube.add_aux_coord(coord, 0)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ('netcdf',
                                      'uint32_auxiliary_coord_netcdf3.cml'),
                           checksum=False)

    def test_uint32_data_netcdf3(self):
        self.cube.data = self.cube.data.astype(np.uint32)
        with self.temp_filename(suffix='.nc') as filename:
            iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')
            reloaded = iris.load_cube(filename)
            self.assertCML(reloaded, ('netcdf',
                                      'uint32_data_netcdf3.cml'))

    def test_uint64_dimension_coord_netcdf3(self):
        # Points that cannot be safely cast to int32.
        coord = iris.coords.DimCoord(np.array([0, 18446744073709551615],
                                              dtype=np.uint64),
                                     long_name='x')
        self.cube.add_dim_coord(coord, 0)
        with self.temp_filename(suffix='.nc') as filename:
            with self.assertRaises(ValueError):
                iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')

    def test_uint64_auxiliary_coord_netcdf3(self):
        # Points that cannot be safely cast to int32.
        coord = iris.coords.AuxCoord(np.array([0, 18446744073709551615],
                                              dtype=np.uint64),
                                     long_name='x')
        self.cube.add_aux_coord(coord, 0)
        with self.temp_filename(suffix='.nc') as filename:
            with self.assertRaises(ValueError):
                iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')

    def test_uint64_data_netcdf3(self):
        # Data that cannot be safely cast to int32.
        self.cube.data = self.cube.data.astype(np.uint64)
        self.cube.data[0, 1] = 18446744073709551615
        with self.temp_filename(suffix='.nc') as filename:
            with self.assertRaises(ValueError):
                iris.save(self.cube, filename, netcdf_format='NETCDF3_CLASSIC')


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
            self.assertTrue(len(cube.attributes["ukmo__process_flags"]) == 1,
                            "Mismatch in number of process flags.")
            process_flag = cube.attributes["ukmo__process_flags"][0]
            self.assertEquals(process_flag, process_desc)

            os.remove(temp_filename)

        # Test mutiple process flags
        multiple_bit_values = ((128, 64), (4096, 1024), (8192, 1024))

        # Maps lbproc value to the process flags that should be created
        multiple_map = {bits: [iris.fileformats.pp.lbproc_map[bit] for
                               bit in bits] for bits in multiple_bit_values}

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
            self.assertTrue(len(process_flags) == len(bits), 'Mismatch in '
                            'number of process flags.')
            self.assertEquals(set(process_flags), set(descriptions))

            os.remove(temp_filename)


if __name__ == "__main__":
    tests.main()
