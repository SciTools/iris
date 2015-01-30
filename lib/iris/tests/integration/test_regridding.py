# (C) British Crown Copyright 2013 - 2015, Met Office
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

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import iris.quickplot as qplt


@tests.skip_data
@tests.skip_plot
class TestOSGBToLatLon(tests.GraphicsTest):
    def setUp(self):
        path = tests.get_data_path(
            ('NIMROD', 'uk2km', 'WO0000000003452',
             '201007020900_u1096_ng_ey00_visibility0180_screen_2km'))
        self.src = iris.load_cube(path)[0]
        self.src.data = self.src.data.astype(np.float32)
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
        qplt.pcolor(result, antialiased=False)
        qplt.plt.gca().coastlines()

    def test_linear(self):
        self._regrid('linear')
        self.check_graphic()

    def test_nearest(self):
        self._regrid('nearest')
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestGlobalSubsample(tests.GraphicsTest):
    def setUp(self):
        self.src = global_pp()
        # Subsample and shift the target grid so that we can see a visual
        # difference between regridding scheme methods.
        grid = self.src[1::2, 1::3]
        grid.coord('latitude').points = grid.coord('latitude').points + 1
        grid.coord('longitude').points = grid.coord('longitude').points + 1
        self.grid = grid

    def _regrid(self, method):
        regridder = Regridder(self.src, self.grid, method, 'mask')
        result = regridder(self.src)
        qplt.pcolormesh(result)
        qplt.plt.gca().coastlines()

    def test_linear(self):
        self._regrid('linear')
        self.check_graphic()

    def test_nearest(self):
        self._regrid('nearest')
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
