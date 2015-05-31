# (C) British Crown Copyright 2010 - 2015, Met Office
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
"""Unit tests for the `iris.fileformats.netcdf.load_cubes` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import netCDF4 as nc
import numpy as np

from iris.fileformats.netcdf import load_cubes


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


class TestRotatedLatlons(tests.IrisTest):
    def test_full_pole(self):
        # Check load of Rotated data with specific grid parameters.
        # create a temporary netcdf file + read a cube from it
        with self.temp_filename(suffix='.nc') as temp_ncpath:
            with nc.Dataset(temp_ncpath, 'w') as ds:
                nx, ny = 6, 4
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                _write_nc_var(ds, 'x', dims=('x'),
                              data=np.linspace(0.0, 100.0, nx),
                              attributes=
                                  {'units': 'degrees',
                                   'standard_name': 'grid_longitude'})
                _write_nc_var(ds, 'y', dims=('y'),
                              data=np.linspace(0.0, 70.0, ny),
                              attributes=
                                  {'units': 'degrees',
                                   'standard_name': 'grid_latitude'})
                _write_nc_var(
                    ds, 'grid_map_var',
                    attributes={
                        'grid_mapping_name': 'rotated_latitude_longitude',
                        'grid_north_pole_latitude': 32.5,
                        'grid_north_pole_longitude': 170.0})
                _write_nc_var(ds, 'temperature', dims=('y', 'x'),
                              data=np.zeros((ny, nx)),
                              attributes={'units': 'K',
                                          'standard_name': 'air_temperature',
                                          'grid_mapping': 'grid_map_var'})
            # load file as a single cube
            cube, = load_cubes([temp_ncpath])

        # test cube properties as required
        x_coord = cube.coord(axis='x')
        self.assertEqual(x_coord.name(), 'grid_longitude')
        cs = x_coord.coord_system
        self.assertEqual(cs.grid_mapping_name, 'rotated_latitude_longitude')
        self.assertEqual(cs.grid_north_pole_latitude, 32.5)
        self.assertEqual(cs.grid_north_pole_longitude, 170.0)

    def test_default_pole(self):
        # Check load of Rotated data with no (=default) pole parameters
        # -- which is thus equivalent to standard lat-lon

        # create a temporary netcdf file + read a cube from it
        with self.temp_filename(suffix='.nc') as temp_ncpath:
            with nc.Dataset(temp_ncpath, 'w') as ds:
                nx, ny = 6, 4
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                _write_nc_var(ds, 'x', dims=('x'),
                              data=np.linspace(0.0, 100.0, nx),
                              attributes=
                                  {'units': 'degrees',
                                   'standard_name': 'grid_longitude'})
                _write_nc_var(ds, 'y', dims=('y'),
                              data=np.linspace(0.0, 70.0, ny),
                              attributes=
                                  {'units': 'degrees',
                                   'standard_name': 'grid_latitude'})
                _write_nc_var(
                    ds, 'grid_map_var',
                    attributes={
                        'grid_mapping_name': 'rotated_latitude_longitude'})
                _write_nc_var(ds, 'temperature', dims=('y', 'x'),
                              data=np.zeros((ny, nx)),
                              attributes={'units': 'K',
                                          'standard_name': 'air_temperature',
                                          'grid_mapping': 'grid_map_var'})
            # load file as a single cube
            cube, = load_cubes([temp_ncpath])

        # test cube properties as required
        x_coord = cube.coord(axis='x')
        self.assertEqual(x_coord.name(), 'grid_longitude')
        cs = x_coord.coord_system
        self.assertEqual(cs.grid_mapping_name, 'rotated_latitude_longitude')
        self.assertEqual(cs.grid_north_pole_latitude, 90.0)
        self.assertEqual(cs.grid_north_pole_longitude, 0.0)


class TestLambertConformal(tests.IrisTest):
    def _test_load_lambert_conformal(self, parallels, centre_latlons,
                                     expect_no_cs=False):
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
                cs_attrs = {'grid_mapping_name': 'lambert_conformal_conic'}
                if parallels:
                    cs_attrs['standard_parallel'] = parallels
                if central_lat is not None:
                    cs_attrs['latitude_of_projection_origin'] = central_lat
                if central_lon is not None:
                    cs_attrs['longitude_of_central_meridian'] = central_lon
                _write_nc_var(ds, 'Lambert_Conformal', attributes=cs_attrs)
                _write_nc_var(ds, 'temperature', dims=('y', 'x'),
                              data=np.zeros((ny, nx)),
                              attributes={'units': 'K',
                                          'standard_name': 'air_temperature',
                                          'grid_mapping': 'Lambert_Conformal'})
            # load file as a single cube
            cube, = load_cubes([temp_ncpath])

        # test cube properties as required
        x_coord = cube.coord(axis='x')
        self.assertEqual(x_coord.name(), 'projection_x_coordinate')
        cs = getattr(x_coord, 'coord_system', None)
        if expect_no_cs:
            self.assertIsNone(cs)
        else:
            self.assertEqual(cs.grid_mapping_name, 'lambert_conformal')
            self.assertEqual(cs.central_lat, central_lat)
            self.assertEqual(cs.central_lon, central_lon)
            self.assertArrayAllClose(cs.standard_parallels, parallels)

    def test_single_parallel(self):
        # Check Lambert Conformal load with a single parallel ("1SP").
        self._test_load_lambert_conformal(
            parallels=(50.0,),
            centre_latlons=(50.0, 107.0))

    def test_two_parallels(self):
        # Check Lambert Conformal load with two parallels ("2SP").
        self._test_load_lambert_conformal(
            parallels=(-35.3, -73.0),
            centre_latlons=(-51.2, 137.0))

    def test_no_parallels(self):
        # Check Lambert Conformal load with no parallels gets NO cs.
        self._test_load_lambert_conformal(
            parallels=None,
            centre_latlons=(-51.2, 137.0),
            expect_no_cs=True)

    def test_no_centre_lat(self):
        # Check Lambert Conformal load with no central lat gets NO cs.
        self._test_load_lambert_conformal(
            parallels=(-35.3, -73.0),
            centre_latlons=(None, 137.0),
            expect_no_cs=True)

    def test_no_centre_lon(self):
        # Check Lambert Conformal load with no central lon gets NO cs.
        self._test_load_lambert_conformal(
            parallels=(-35.3, -73.0),
            centre_latlons=(-51.2, None),
            expect_no_cs=True)


class TestPlainLatLons(tests.IrisTest):

    def test_no_ellipsoid(self):
        # check for a 'plain' grid_mapping var with no ellipsoid information.
        with self.temp_filename(suffix='.nc') as temp_ncpath:
            with nc.Dataset(temp_ncpath, 'w') as ds:
                nx, ny = 6, 4
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                _write_nc_var(ds, 'x', dims=('x'),
                              data=np.linspace(0.0, 100.0, nx),
                              attributes=
                                  {'units': 'degrees_east',
                                   'standard_name': 'longitude'})
                _write_nc_var(ds, 'y', dims=('y'),
                              data=np.linspace(0.0, 100.0, ny),
                              attributes=
                                  {'units': 'degrees_north',
                                   'standard_name': 'latitude'})
                _write_nc_var(
                    ds, 'grid_mapping_latlon',
                    attributes={'grid_mapping_name': 'latitude_longitude'})
                _write_nc_var(
                    ds, 'temperature', dims=('y', 'x'),
                    data=np.zeros((ny, nx)),
                    attributes={'units': 'K',
                                'standard_name': 'air_temperature',
                                'grid_mapping': 'grid_mapping_latlon'})
            # load file as a single cube
            cube, = load_cubes([temp_ncpath])

        # test resulting cube properties
        x_coord = cube.coord(axis='x')
        self.assertEqual(x_coord.name(), 'longitude')
        self.assertIsNone(x_coord.coord_system)

    def test_no_grid_mapping(self):
        # check that a 'plain' latlon grid has no ellipsoid.
        with self.temp_filename(suffix='.nc') as temp_ncpath:
            with nc.Dataset(temp_ncpath, 'w') as ds:
                nx, ny = 6, 4
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                _write_nc_var(ds, 'x', dims=('x'),
                              data=np.linspace(0.0, 100.0, nx),
                              attributes=
                                  {'units': 'degrees_east',
                                   'standard_name': 'longitude'})
                _write_nc_var(ds, 'y', dims=('y'),
                              data=np.linspace(0.0, 100.0, ny),
                              attributes=
                                  {'units': 'degrees_north',
                                   'standard_name': 'latitude'})
                _write_nc_var(
                    ds, 'temperature', dims=('y', 'x'),
                    data=np.zeros((ny, nx)),
                    attributes={'units': 'K',
                                'standard_name': 'air_temperature'})
            # load file as a single cube
            cube, = load_cubes([temp_ncpath])

        # test resulting cube properties
        x_coord = cube.coord(axis='x')
        self.assertEqual(x_coord.name(), 'longitude')
        self.assertIsNone(x_coord.coord_system)


if __name__ == "__main__":
    tests.main()
