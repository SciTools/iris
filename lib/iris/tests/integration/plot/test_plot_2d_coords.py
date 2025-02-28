# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test plots with two dimensional coordinates."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import iris
from iris.analysis.cartography import unrotate_pole
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.tests import _shared_utils

# Run tests in no graphics mode if matplotlib is not available.
if _shared_utils.MPL_AVAILABLE:
    import iris.quickplot as qplt


@_shared_utils.skip_data
def simple_cube_w_2d_coords():
    path = _shared_utils.get_data_path(("NetCDF", "ORCA2", "votemper.nc"))
    cube = iris.load_cube(path)
    return cube


@_shared_utils.skip_plot
@_shared_utils.skip_data
class Test(_shared_utils.GraphicsTest):
    def test_2d_coord_bounds_platecarree(self):
        # To avoid a problem with Cartopy smearing the data where the
        #  longitude wraps, we set the central_longitude.
        #  SciTools/cartopy#1421
        cube = simple_cube_w_2d_coords()[0, 0]
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        qplt.pcolormesh(cube)

        # Cartopy can't reliably set y-limits with curvilinear plotting.
        #  SciTools/cartopy#2121
        y_lims = [m(cube.coord("latitude").points) for m in (np.min, np.max)]
        ax.set_ylim(*y_lims)

        ax.coastlines(resolution="110m", color="red")
        self.check_graphic()

    def test_2d_coord_bounds_northpolarstereo(self):
        cube = simple_cube_w_2d_coords()[0, 0]
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        qplt.pcolormesh(cube)
        ax.coastlines(resolution="110m", color="red")
        self.check_graphic()


@_shared_utils.skip_plot
class Test2dContour(_shared_utils.GraphicsTest):
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
