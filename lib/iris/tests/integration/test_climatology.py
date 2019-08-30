# (C) British Crown Copyright 2014 - 2017, Met Office
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
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from datetime import datetime

import netCDF4 as nc
import numpy as np

import iris
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube


class TestClimatology(iris.tests.IrisTest):
    @staticmethod
    def _simple_data_cube():

        def jan_offset(day, year):
            dt = (datetime(year, 1, day) - datetime(1970, 1, 1))
            return dt.total_seconds() / (24. * 3600)


        days = range(10, 15)
        years = [[year, year + 10] for year in [2001] * 4]
        days_since = [[jan_offset(day, yr1), jan_offset(day, yr2)]
                      for (day, [yr1, yr2])
                      in zip(days, years)]
        time_bounds = np.array(days_since)
        time_points = time_bounds[..., 0]

        lon = np.linspace(-25, 25, 5)
        lat = np.linspace(0, 60, 3)

        time_dim = DimCoord(time_points,
                            standard_name='time',
                            bounds=time_bounds,
                            units='days since 1970-01-01 00:00:00-00')
        lon_dim = DimCoord(lon, standard_name='longitude')
        lat_dim = DimCoord(lat, standard_name='latitude')

        data_shape = (len(time_points), len(lat), len(lon))
        values = np.zeros(shape=data_shape, dtype=np.int8)
        cube = Cube(values)
        cube.add_dim_coord(time_dim, 0)
        cube.add_dim_coord(lat_dim, 1)
        cube.add_dim_coord(lon_dim, 2)
        cube.rename('climatology test')
        cube.units = 'Kelvin'
        cube.add_cell_method(CellMethod('mean over years', coords='time'))

        return cube

    @staticmethod
    def _bounds_to_climatology(filepath):
        # Hack file until Iris does it right ..
        ds = nc.Dataset(filepath, 'r+')
        ds.variables['time'].renameAttribute('bounds', 'climatology')
        old_name = 'time_bnds'
        new_name = 'time_climatology'
        assert (ds.variables['time'].climatology == old_name)
        ds.variables['time'].climatology = new_name
        ds.renameVariable(old_name, new_name)
        ds.close()

    def test_save_simpledata(self):
        cube = self._simple_data_cube()
        # Write Cube to netCDF file.
        with self.temp_filename(suffix='.nc') as filepath_out:
            iris.save(cube, filepath_out)
            self._bounds_to_climatology(filepath_out)
            # Check CDL of saved result.
            self.assertCDL(filepath_out, flags='')

    def test_cube_to_cube(self):
        cube_out = self._simple_data_cube()
        bounds_out = cube_out.coord('time').bounds
        # Write Cube to netCDF file.
        with self.temp_filename(suffix='.nc') as filepath_out:
            iris.save(cube_out, filepath_out)
            self._bounds_to_climatology(filepath_out)
            cube_in = iris.load_cube(filepath_out)
            bounds_in = cube_in.coord('time').bounds
            self.assertArrayEqual(bounds_in, bounds_out)


if __name__ == "__main__":
    tests.main()
