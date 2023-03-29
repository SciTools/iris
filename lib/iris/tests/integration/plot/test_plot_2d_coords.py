# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test plots with two dimensional coordinates.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import iris
from iris.analysis.cartography import unrotate_pole
from iris.coords import AuxCoord
from iris.cube import Cube

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import iris.quickplot as qplt


@tests.skip_data
def simple_cube_w_2d_coords():
    path = tests.get_data_path(("NetCDF", "ORCA2", "votemper.nc"))
    cube = iris.load_cube(path)
    return cube


@tests.skip_plot
@tests.skip_data
class Test(tests.GraphicsTest):
    def test_2d_coord_bounds_platecarree(self):
        # To avoid a problem with Cartopy smearing the data where the
        # longitude wraps, we set the central_longitude
        cube = simple_cube_w_2d_coords()[0, 0]
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        qplt.pcolormesh(cube)
        ax.coastlines(resolution="110m", color="red")
        self.check_graphic()

    def test_2d_coord_bounds_northpolarstereo(self):
        cube = simple_cube_w_2d_coords()[0, 0]
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        qplt.pcolormesh(cube)
        ax.coastlines(resolution="110m", color="red")
        self.check_graphic()


@tests.skip_plot
class Test2dContour(tests.GraphicsTest):
    def test_2d_coords_contour(self):
        ny, nx = 4, 6
        x1 = np.linspace(-20, 70, nx)
        y1 = np.linspace(10, 60, ny)
        data = np.zeros((ny, nx))
        data.flat[:] = np.arange(nx * ny) % 7
        cube = Cube(data, long_name="Odd data")
        x2, y2 = np.meshgrid(x1, y1)
        true_lons, true_lats = unrotate_pole(x2, y2, -130.0, 77.0)
        co_x = AuxCoord(true_lons, standard_name="longitude", units="degrees")
        co_y = AuxCoord(true_lats, standard_name="latitude", units="degrees")
        cube.add_aux_coord(co_y, (0, 1))
        cube.add_aux_coord(co_x, (0, 1))
        ax = plt.axes(projection=ccrs.PlateCarree())
        qplt.contourf(cube)
        ax.coastlines(resolution="110m", color="red")
        ax.gridlines(draw_labels=True)
        ax.set_extent((0, 180, 0, 90))
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
