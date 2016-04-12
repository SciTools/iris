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
"""
Integration test for loading netcdf data with a grid mapping but no ellipsoid.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import shutil
import subprocess

import netCDF4 as nc

from iris import load_cube


def _create_no_ellipsoid_testfile(problem_filepath, grid_mapping_name):
    # Make a copy of the sample datafile.
    test_filepath = tests.get_data_path(
        ('NetCDF', 'testing', 'small_theta_colpex.nc'))
    shutil.copy(test_filepath, problem_filepath)

    # Open for modifying with netCDF4.
    ds = nc.Dataset(problem_filepath, 'a')

    # Get the main and grid-mapping vars.
    main_var_name = 'air_potential_temperature'
    main_var = ds.variables[main_var_name]
    gmv_original_name = main_var.grid_mapping
    # Rename the grid-mapping variable to a more neutral name.
    gmv_new_name = 'crs'
    ds.renameVariable(gmv_original_name, gmv_new_name)
    # Rename the main-var reference to it.
    main_var.grid_mapping = gmv_new_name

    # Modify the grid mapping variable :
    # remove all attributes, and redefine the grid mapping as plain-lat-long.
    gmv = ds.variables[gmv_new_name]
    for attr_name in gmv.ncattrs():
        gmv.delncattr(attr_name)
    gmv.grid_mapping_name = grid_mapping_name

    # Save.
    ds.close()


class MixinCheckNoEllipsoid(object):
    grid_mapping_name = None

    def test_nc_load_no_ellipsoid(self):
        with self.temp_filename(suffix='.nc') as temp_filepath:
            _create_no_ellipsoid_testfile(
                temp_filepath, grid_mapping_name=self.grid_mapping_name)
            result = load_cube(temp_filepath)
            coord_system = result.coord_system()
            self.assertIsNone(coord_system.ellipsoid, None)


class TestRotatedNoEllipsoid(MixinCheckNoEllipsoid, tests.IrisTest):
    grid_mapping_name = 'rotated_latitude_longitude'


class TestUnrotatedNoEllipsoid(MixinCheckNoEllipsoid, tests.IrisTest):
    grid_mapping_name = 'latitude_longitude'


if __name__ == '__main__':
    tests.main()
