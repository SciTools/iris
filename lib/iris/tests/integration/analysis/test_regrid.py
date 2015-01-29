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
class Test___call____visual(tests.GraphicsTest):
    def setUp(self):
        # Regridder method and extrapolation-mode.
        self.args = ('linear', 'mask')

    def test_osgb_to_latlon(self):
        path = tests.get_data_path(
            ('NIMROD', 'uk2km', 'WO0000000003452',
             '201007020900_u1096_ng_ey00_visibility0180_screen_2km'))
        src = iris.load_cube(path)[0]
        src.data = src.data.astype(np.float32)
        grid = Cube(np.empty((73, 96)))
        cs = GeogCS(6370000)
        lat = DimCoord(np.linspace(46, 65, 73), 'latitude', units='degrees',
                       coord_system=cs)
        lon = DimCoord(np.linspace(-14, 8, 96), 'longitude', units='degrees',
                       coord_system=cs)
        grid.add_dim_coord(lat, 0)
        grid.add_dim_coord(lon, 1)
        regridder = Regridder(src, grid, *self.args)
        result = regridder(src)
        qplt.pcolor(result, antialiased=False)
        qplt.plt.gca().coastlines()
        self.check_graphic()

    def test_subsample(self):
        src = global_pp()
        grid = src[::2, ::3]
        regridder = Regridder(src, grid, *self.args)
        result = regridder(src)
        qplt.pcolormesh(result)
        qplt.plt.gca().coastlines()
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
