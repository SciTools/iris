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
"""Unit tests for the `iris.fileformats.netcdf.save` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import netCDF4 as nc
import numpy as np

from iris.cube import Cube
from iris.fileformats.netcdf import save, CF_CONVENTIONS_VERSION


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


if __name__ == "__main__":
    tests.main()
