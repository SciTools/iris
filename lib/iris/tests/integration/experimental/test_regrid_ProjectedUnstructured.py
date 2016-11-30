# (C) British Crown Copyright 2016, Met Office
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
"""Integration tests for experimental regridding."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import cartopy.crs as ccrs
from cf_units import Unit
import numpy as np

import iris
from iris.coord_systems import GeogCS
from iris.tests.stock import global_pp, simple_3d
from iris.experimental.regrid import (ProjectedUnstructuredNearest,
                                      ProjectedUnstructuredLinear)


@tests.skip_data
class TestProjectedUnstructured(tests.IrisTest):
    def setUp(self):
        path = tests.get_data_path(('NetCDF', 'unstructured_grid',
                                   'theta_nodal_xios.nc'))
        self.src = iris.load_cube(path, 'Potential Temperature')

        src_lat = self.src.coord('latitude')
        src_lon = self.src.coord('longitude')
        src_lat.coord_system = src_lon.coord_system = GeogCS(6370000)
        src_lat.convert_units(Unit('degrees'))
        src_lon.convert_units(Unit('degrees'))

        self.grid = simple_3d()[0, :, :]
        self.grid.coord('latitude').coord_system = GeogCS(6370000)
        self.grid.coord('longitude').coord_system = GeogCS(6370000)

    def test_nearest(self):
        res = self.src.regrid(self.grid, ProjectedUnstructuredNearest())
        self.assertArrayShapeStats(res, (1, 6, 3, 4), 315.8906568, 11.00072015)

    def test_nearest_platecarree(self):
        crs = ccrs.PlateCarree()
        res = self.src.regrid(self.grid, ProjectedUnstructuredNearest(crs))
        self.assertArrayShapeStats(res, (1, 6, 3, 4), 315.8906481, 11.00072015)

    def test_linear(self):
        res = self.src.regrid(self.grid, ProjectedUnstructuredLinear())
        self.assertArrayShapeStats(res, (1, 6, 3, 4), 315.890713787, 11.000729)


if __name__ == "__main__":
    tests.main()
