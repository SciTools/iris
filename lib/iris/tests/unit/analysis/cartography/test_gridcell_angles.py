# (C) British Crown Copyright 2015 - 2016, Met Office
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
Unit tests for the function
:func:`iris.analysis.cartography.gridcell_angles`.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

import cartopy.crs as ccrs
from iris.cube import Cube
from iris.coords import DimCoord, AuxCoord
import iris.coord_systems
from iris.analysis.cartography import unrotate_pole

from iris.analysis.cartography import (gridcell_angles,
                                       rotate_grid_vectors)


def _rotated_grid_sample(pole_lat=15, pole_lon=-180,
                         lon_bounds=np.linspace(-30, 30, 8, endpoint=True),
                         lat_bounds=np.linspace(-30, 30, 8, endpoint=True)):
    # Calculate *true* lat_bounds+lon_bounds for the rotated grid.
    lon_bounds = np.array(lon_bounds, dtype=float)
    lat_bounds = np.array(lat_bounds, dtype=float)
    # Construct centrepoints.
    lons = 0.5 * (lon_bounds[:-1] + lon_bounds[1:])
    lats = 0.5 * (lat_bounds[:-1] + lat_bounds[1:])
    # Convert all to full 2d arrays.
    lon_bounds, lat_bounds = np.meshgrid(lon_bounds, lat_bounds)
    lons, lats = np.meshgrid(lons, lats)
    # Calculate true lats+lons for all points.
    lons_true_bds, lats_true_bds = unrotate_pole(lon_bounds, lat_bounds,
                                                 pole_lon, pole_lat)
    lons_true, lats_true = unrotate_pole(lons, lats, pole_lon, pole_lat)
    # Make the 'unified' bounds into contiguous (ny, nx, 4) arrays.
    def expand_unified_bds(bds):
        ny, nx = bds.shape
        bds_4 = np.zeros((ny - 1, nx - 1, 4))
        bds_4[:, :, 0] = bds[:-1, :-1]
        bds_4[:, :, 1] = bds[:-1, 1:]
        bds_4[:, :, 2] = bds[1:, 1:]
        bds_4[:, :, 3] = bds[1:, :-1]
        return bds_4

    lon_true_bds4, lat_true_bds4 = (expand_unified_bds(bds)
                                    for bds in (lons_true_bds, lats_true_bds))
    # Make these into a 2d-latlon grid for a cube
    cube = Cube(np.zeros(lon_true_bds4.shape[:-1]))
    co_x = AuxCoord(lons_true, bounds=lon_true_bds4,
                    standard_name='longitude', units='degrees')
    co_y = AuxCoord(lats_true, bounds=lat_true_bds4,
                    standard_name='latitude', units='degrees')
    cube.add_aux_coord(co_x, (0, 1))
    cube.add_aux_coord(co_y, (0, 1))
    return cube


class TestGridcellAngles(tests.IrisTest):
    def test_values(self):
        # Construct a rotated-pole grid and check angle calculation.
        testcube = _rotated_grid_sample()
        angles_cube = gridcell_angles(testcube)
        angles_cube.convert_units('radians')

        # testing phase...
        print(np.rad2deg(angles_cube.data))
        
        import matplotlib.pyplot as plt
        plt.switch_backend('tkagg')
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0.0,
                                                   central_latitude=90.0,))
        ax.coastlines()
        ax.gridlines()
        for i_bnd in range(4):
            color = ['black', 'red', 'blue', 'magenta'][i_bnd]
            plt.plot(testcube.coord('longitude').bounds[..., i_bnd],
                     testcube.coord('latitude').bounds[..., i_bnd],
                     '+', markersize=10., markeredgewidth=2.,
                     markerfacecolor=color, markeredgecolor=color,
                     transform=ccrs.PlateCarree())


        # Show plain 0,1 + 1,0 vectors unrotated at the given points.
        pts_shape = testcube.coord('longitude').shape
        u0 = np.ones(pts_shape)
        v0 = np.zeros(pts_shape)
        u1 = v0.copy()
        v1 = u0.copy()

#        u0_cube, u1_cube, v0_cube, v1_cube = [testcube.copy(data=aa)
#                                              for aa in (u0, v0, u1, v1)]
#        u0r_cube, v0r_cube = rotate_grid_vectors(
#            u0_cube, v0_cube, grid_angles_cube=angles_cube)

        scale = 4.0e-6
        plt.quiver(testcube.coord('longitude').points,
                   testcube.coord('latitude').points,
                   u0, v0, color='blue', linewidth=0.5, scale_units='xy', scale=scale,
                   transform=ccrs.PlateCarree())
        plt.quiver(testcube.coord('longitude').points,
                   testcube.coord('latitude').points,
                   u1, v1, color='red', linewidth=0.5, scale_units='xy', scale=scale,
                   transform=ccrs.PlateCarree())

#        plt.quiver(testcube.coord('longitude').points,
#                   testcube.coord('latitude').points,
#                   u0r_cube.data, v0r_cube.data,
#                   color='red',
#                   transform=ccrs.PlateCarree())

        # Also draw small lines pointing at the correct angle.
        x0s = testcube.coord('longitude').points
        y0s = testcube.coord('latitude').points
        ny, nx = x0s.shape
        size_degrees = 5.0
        angles = angles_cube.copy()
        angles.convert_units('radians')
        angles = angles.data
        lats = testcube.coord('latitude').copy()
        lats.convert_units('radians')
        lats = lats.points
        dys = size_degrees * np.sin(angles) / np.cos(-lats)
        dxs = size_degrees * np.cos(angles)
        x1s = x0s + dxs
        y1s = y0s + dys
        for iy in range(ny):
            for ix in range(nx):
                plt.plot([x0s[iy, ix], x1s[iy, ix]],
                         [y0s[iy, ix], y1s[iy, ix]],
                         'o-', markersize=4., markeredgewidth=0.,
                         color='green',
                         transform=ccrs.PlateCarree())



        ax.set_global()
        plt.show()


        self.assertArrayAllClose(
            angles_cube.data,
            [[100.0, 100.0, 100.0],
             [100.0, 100.0, 100.0],
             [100.0, 100.0, 100.0]])


if __name__ == "__main__":
    tests.main()
