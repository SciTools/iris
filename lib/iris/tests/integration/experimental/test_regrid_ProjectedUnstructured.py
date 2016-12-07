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
from iris.tests.stock import global_pp
from iris.experimental.regrid import ProjectedUnstructuredNearest


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

        self.global_grid = global_pp()

    def test_nearest(self):
        res = self.src.regrid(self.global_grid, ProjectedUnstructuredNearest())
        self.assertArrayShapeStats(res, (1, 6, 73, 96),
                                   315.8913582, 11.00063766248)

    def test_nearest_sinusoidal(self):
        crs = ccrs.Sinusoidal()
        res = self.src.regrid(self.global_grid,
                              ProjectedUnstructuredNearest(crs))
        self.assertArrayShapeStats(res, (1, 6, 73, 96),
                                   315.891358296, 11.000639227)

    def test_nearest_gnomonic_uk_domain(self):
        crs = ccrs.Gnomonic(central_latitude=60.0)
        uk_grid = self.global_grid.intersection(longitude=(-20, 20),
                                                latitude=(40, 80))
        res = self.src.regrid(uk_grid, ProjectedUnstructuredNearest(crs))
        self.assertArrayShapeStats(res, (1, 6, 17, 11),
                                   315.8873266, 11.0006664668)


if __name__ == "__main__":
    tests.main()
