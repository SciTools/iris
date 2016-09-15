# (C) British Crown Copyright 2013 - 2016, Met Office
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
"""Integration tests for regridding."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.analysis._regrid import RectilinearRegridder as Regridder
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests.stock import global_pp


@tests.skip_data
class TestOSGBToLatLon(tests.IrisTest):
    def setUp(self):
        path = tests.get_data_path(
            ('NIMROD', 'uk2km', 'WO0000000003452',
             '201007020900_u1096_ng_ey00_visibility0180_screen_2km'))
        self.src = iris.load_cube(path)[0]
        # Cast up to float64, to work around numpy<=1.8 bug with means of
        # arrays of 32bit floats.
        self.src.data = self.src.data.astype(np.float64)
        self.grid = Cube(np.empty((73, 96)))
        cs = GeogCS(6370000)
        lat = DimCoord(np.linspace(46, 65, 73), 'latitude', units='degrees',
                       coord_system=cs)
        lon = DimCoord(np.linspace(-14, 8, 96), 'longitude', units='degrees',
                       coord_system=cs)
        self.grid.add_dim_coord(lat, 0)
        self.grid.add_dim_coord(lon, 1)

    def _regrid(self, method):
        regridder = Regridder(self.src, self.grid, method, 'mask')
        result = regridder(self.src)
        return result

    def test_linear(self):
        res = self._regrid('linear')
        self.assertArrayShapeStats(res, (73, 96), -16100.351951, 5603.850769)

    def test_nearest(self):
        res = self._regrid('nearest')
        self.assertArrayShapeStats(res, (73, 96), -16095.965585, 5612.657155)


@tests.skip_data
class TestGlobalSubsample(tests.IrisTest):
    def setUp(self):
        self.src = global_pp()
        _ = self.src.data
        # Cast up to float64, to work around numpy<=1.8 bug with means of
        # arrays of 32bit floats.
        self.src.data = self.src.data.astype(np.float64)
        # Subsample and shift the target grid so that we can see a visual
        # difference between regridding scheme methods.
        grid = self.src[1::2, 1::3]
        grid.coord('latitude').points = grid.coord('latitude').points + 1
        grid.coord('longitude').points = grid.coord('longitude').points + 1
        self.grid = grid

    def _regrid(self, method):
        regridder = Regridder(self.src, self.grid, method, 'mask')
        result = regridder(self.src)
        return result

    def test_linear(self):
        res = self._regrid('linear')
        self.assertArrayShapeStats(res, (36, 32), 280.35907, 15.997223)

    def test_nearest(self):
        res = self._regrid('nearest')
        self.assertArrayShapeStats(res, (36, 32), 280.33726, 16.064001)


if __name__ == "__main__":
    tests.main()
