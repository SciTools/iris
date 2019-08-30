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
from os.path import join as path_join
import shutil
from subprocess import check_call
import tempfile

import netCDF4 as nc
import numpy as np

import iris
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube


class TestClimatology(iris.tests.IrisTest):
    reference_cdl_path = (
        '../results/integration/climatology/TestClimatology/'
        'reference_simpledata.cdl')

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

    @classmethod
    def _simple_cdl_string(cls):
        with open(cls.reference_cdl_path, 'r') as f:
            cdl_content = f.read()
        # Add the expected CDL first line since this is removed from the
        # stored results file.
        cdl_content = 'netcdf {\n' + cdl_content

        return cdl_content

    @staticmethod
    def _bounds_to_climatology(file_path):
        # Hack file until Iris does it right ..
        # TODO: remove this and any references to it.
        ds = nc.Dataset(file_path, 'r+')
        ds.variables['time'].renameAttribute('bounds', 'climatology')
        old_name = 'time_bnds'
        new_name = 'time_climatology'
        assert (ds.variables['time'].climatology == old_name)
        ds.variables['time'].climatology = new_name
        ds.renameVariable(old_name, new_name)
        ds.close()

    @staticmethod
    def _load_sanitised_cube(filepath):
        cube = iris.load_cube(filepath)
        # Remove attributes convention, if any.
        cube.attributes.pop('Conventions', None)
        # Remove any var-names.
        for coord in cube.coords():
            coord.var_name = None
        cube.var_name = None
        return cube

    @classmethod
    def setUpClass(cls):
        # Create a temp directory for temp files.
        cls.temp_dir = tempfile.mkdtemp()
        cls.path_ref_cdl = path_join(cls.temp_dir, 'standard.cdl')
        cls.path_ref_nc = path_join(cls.temp_dir, 'standard.nc')
        # Create reference CDL file.
        with open(cls.path_ref_cdl, 'w') as f_out:
            f_out.write(cls._simple_cdl_string())
        # Create reference netCDF file from reference CDL.
        command = 'ncgen -o {} {}'.format(
            cls.path_ref_nc, cls.path_ref_cdl)
        check_call(command, shell=True)
        cls.path_temp_nc = path_join(cls.temp_dir, 'tmp.nc')

        # Create reference cube.
        cls.cube_ref = cls._simple_data_cube()

    @classmethod
    def tearDownClass(cls):
        # Destroy a temp directory for temp files.
        shutil.rmtree(cls.temp_dir)

###############################################################################

    def test_save_simpledata(self):
        # Create file from cube, test against reference CDL.
        cube = self.cube_ref
        iris.save(cube, self.path_temp_nc)
        self._bounds_to_climatology(self.path_temp_nc)
        self.assertCDL(
            self.path_temp_nc,
            reference_filename=self.reference_cdl_path,
            flags='')

    def test_load_simpledata(self):
        # Create cube from file, test against reference cube.
        cube = self._load_sanitised_cube(self.path_ref_nc)
        self.assertEqual(cube, self.cube_ref)

    def test_cube_to_cube(self):
        # Save reference cube to file, load cube from same file, test against
        # reference cube.
        iris.save(self.cube_ref, self.path_temp_nc)
        cube = self._load_sanitised_cube(self.path_temp_nc)
        self.assertEqual(cube, self.cube_ref)

    def test_file_to_file(self):
        # Load cube from reference file, save same cube to file, test against
        # reference CDL.
        cube = iris.load_cube(self.path_ref_nc)
        iris.save(cube, self.path_temp_nc)
        self._bounds_to_climatology(self.path_temp_nc)
        self.assertCDL(
            self.path_temp_nc,
            reference_filename=self.reference_cdl_path,
            flags='')


if __name__ == "__main__":
    tests.main()
