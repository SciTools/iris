# (C) British Crown Copyright 2018, Met Office
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

import matplotlib.pyplot as plt
from orca_utils.plot_testing.blockplot_from_bounds import blockplot_2dll


def _rotated_grid_sample(pole_lat=15, pole_lon=-180,
                         lon_bounds=np.linspace(-30, 30, 6, endpoint=True),
                         lat_bounds=np.linspace(-30, 30, 6, endpoint=True)):
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
    def _singlecell_30deg_cube(self, x0=90., y0=0., dx=20., dy=10.):
        x_pts = np.array([[x0]])
        y_pts = np.array([[y0]])
        x_bds = x0 + dx * np.array([[[-1., 1, 0.5, -1.5]]])
#        self.assertArrayAllClose(x_bds, np.array([[[70., 110, 100, 60]]]))
        y_bds = y0 + dy * np.array([[[-1., 1, 3, 1]]])
#        self.assertArrayAllClose(y_bds, np.array([[[-10., 10, 30, 10]]]))
        co_x = AuxCoord(points=x_pts, bounds=x_bds,
                        standard_name='longitude', units='degrees')
        co_y = AuxCoord(points=y_pts, bounds=y_bds,
                        standard_name='latitude', units='degrees')
        cube = Cube(np.zeros((1, 1)))
        cube.add_aux_coord(co_x, (0, 1))
        cube.add_aux_coord(co_y, (0, 1))
        return cube

    def _singlecell_diamond_cube(self, x0=90., y0=0., dy=10., dx_eq=None):
        if dx_eq is None:
            dx_eq = dy
        x_pts = np.array([[x0]])
        y_pts = np.array([[y0]])
        dx = dx_eq / np.cos(np.deg2rad(y0))
        x_bds = np.array([[[x0, x0 + dx, x0, x0 - dx]]])
        y_bds = np.array([[[y0 - dy, y0, y0 + dy, y0]]])
        co_x = AuxCoord(points=x_pts, bounds=x_bds,
                        standard_name='longitude', units='degrees')
        co_y = AuxCoord(points=y_pts, bounds=y_bds,
                        standard_name='latitude', units='degrees')
        cube = Cube(np.zeros((1, 1)))
        cube.add_aux_coord(co_x, (0, 1))
        cube.add_aux_coord(co_y, (0, 1))
        return cube

    def _check_orientations_and_latitudes(self,
                                          method='mid-lhs, mid-rhs',
                                          atol_degrees=0.005,
                                          cellsize_degrees=1.0):
        ny, nx = 7, 9
        x0, x1 = -164, 164
        y0, y1 = -75, 75
        lats = np.linspace(y0, y1, ny, endpoint=True)
        angles = np.linspace(x0, x1, nx, endpoint=True)
        x_pts_2d, y_pts_2d = np.meshgrid(angles, lats)

        # Make gridcells rectangles surrounding these centrepoints, but also
        # tilted at various angles (= same as x-point lons, as that's easy).
#        dx = cellsize_degrees  # half-width of gridcells, in degrees
#        dy = dx   # half-height of gridcells, in degrees

        # Calculate centrepoint lons+lats : in radians, and shape (ny, nx, 1).
        xangs, yangs = np.deg2rad(x_pts_2d), np.deg2rad(y_pts_2d)
        xangs, yangs = [arr[..., None] for arr in (xangs, yangs)]
        # Program which corners are up+down on each gridcell axis.
        dx_corners = [[[-1, 1, 1, -1]]]
        dy_corners = [[[-1, -1, 1, 1]]]
        # Calculate the relative offsets in x+y at the 4 corners.
        x_ofs_2d = cellsize_degrees * np.cos(xangs) * dx_corners
        x_ofs_2d -= cellsize_degrees * np.sin(xangs) * dy_corners
        y_ofs_2d = cellsize_degrees * np.cos(xangs) * dy_corners
        y_ofs_2d += cellsize_degrees * np.sin(xangs) * dx_corners
        # Apply a latitude stretch to make correct angles on the globe.
        y_ofs_2d *= np.cos(yangs)
        # Make bounds arrays by adding the corner offsets to the centrepoints.
        x_bds_2d = x_pts_2d[..., None] + x_ofs_2d
        y_bds_2d = y_pts_2d[..., None] + y_ofs_2d

        # Create a cube with these points + bounds in its 'X' and 'Y' coords.
        co_x = AuxCoord(points=x_pts_2d, bounds=x_bds_2d,
                        standard_name='longitude', units='degrees')
        co_y = AuxCoord(points=y_pts_2d, bounds=y_bds_2d,
                        standard_name='latitude', units='degrees')
        cube = Cube(np.zeros((ny, nx)))
        cube.add_aux_coord(co_x, (0, 1))
        cube.add_aux_coord(co_y, (0, 1))

        # Calculate gridcell angles at each point.
        angles_cube = gridcell_angles(cube, cell_angle_boundpoints=method)

        # Check that the results are a close match to the original intended
        # gridcell orientation angles.
        # NOTE: neither the above gridcell construction nor the calculation
        # itself are exact :  Errors scale as the square of gridcell sizes.
        angles_cube.convert_units('degrees')
        angles_calculated = angles_cube.data
        # Note: expand the original 1-d test angles into the full result shape,
        # just to please 'np.testing.assert_allclose', which doesn't broadcast.
        angles_expected = np.zeros(angles_cube.shape)
        angles_expected[:] = angles
        self.assertArrayAllClose(angles_calculated, angles_expected,
                                 atol=atol_degrees)
        return angles_calculated, angles_expected

    def test_all_orientations_and_latitudes(self):
        self._check_orientations_and_latitudes()

    def test_different_methods(self):
        # Get results with both calculation methods.
        # A smallish cellsize should yield similar results in both cases.
        r1, _ = self._check_orientations_and_latitudes(
            method='mid-lhs, mid-rhs',
            cellsize_degrees=0.1, atol_degrees=0.1)
        r2, _ = self._check_orientations_and_latitudes(
            method='lower-left, lower-right',
            cellsize_degrees=0.1, atol_degrees=0.1)

        print(np.round(r1 - r2, 3))
        for fn in (np.max, np.mean):
            print(fn.__name__, fn(np.abs(r1 - r2)))

        atol = 0.1  # A whole degree - significantly different at higher latitudes.
        self.assertArrayAllClose(r1, r2, atol=atol)

    def test_methods_and_cellsizes(self):
        for cellsize in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            r_mid, exp_mid = self._check_orientations_and_latitudes(
                method='mid-lhs, mid-rhs',
                cellsize_degrees=cellsize, atol_degrees=25)
            r_btm, exp_btm = self._check_orientations_and_latitudes(
                method='lower-left, lower-right',
                cellsize_degrees=cellsize, atol_degrees=25)
            wc_mid = np.max(np.abs(r_mid - exp_mid))
            wc_btm = np.max(np.abs(r_btm - exp_btm))
            msg = ('Cell size = {:5.2f} degrees, wc-abs-errors : '
                   'mid-lr={:7.3f} lower-lr={:7.3f}')
            print(msg.format(cellsize, wc_mid, wc_btm))


#    def test_single_cell_equatorial(self):
#        plt.switch_backend('tkagg')
#        plt.figure(figsize=(10,10))
##        ax = plt.axes(projection=ccrs.Mercator())
##        ax = plt.axes(projection=ccrs.NorthPolarStereo())
#        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=90.,
#                                              central_latitude=30.))
#
#        lon0 = 90.0
#        dy = 1.0
#        dx = 3.0
#        y_0, y_n, ny = -80, 80, 9
#        angles = []
#        for lat in np.linspace(y_0, y_n, ny):
#            cube = self._singlecell_diamond_cube(x0=lon0, y0=lat,
#                                                 dy=dy, dx_eq=dx)
#            angles_cube = gridcell_angles(cube,
##                                          cell_angle_boundpoints='mid-lhs, mid-rhs')
#                                          cell_angle_boundpoints='lower-left, lower-right')
#            tmp_cube = angles_cube.copy()
#            tmp_cube.convert_units('degrees')
##            print('')
##            print(lat)
##            co_x, co_y = (cube.coord(axis=ax) for ax in ('x', 'y'))
##            print()
##            print('  at : {}, {}'.format(co_x.points[0, 0], co_y.points[0, 0]))
##            print('  x-bds:')
##            print(co_x.bounds)
##            print('  y-bds:')
##            print(co_y.bounds)
#            angle = tmp_cube.data[0, 0]
#            angles.append(angle)
#            print(lat, angle)
#            blockplot_2dll(cube)
#
#        ax.coastlines()
#        ax.set_global()
#
#        # Plot constant NEly (45deg) arrows.
#        xx = np.array([lon0] * ny)
#        yy = np.linspace(y_0, y_n, ny) - dy
#        uu = np.array([1.0] * ny)
#        plt.quiver(xx, yy,
#                   uu, np.cos(np.deg2rad(yy)),
#                   zorder=2, color='red',
##                   scale_units='xy',
#                   angles='uv',
#                   transform=ccrs.PlateCarree())
#
#        # Also plot returned angles.
#        angles_arr_rad = np.deg2rad(angles)
#        u_arr = uu * np.cos(angles_arr_rad)
#        v_arr = uu * np.sin(angles_arr_rad) * np.cos(np.deg2rad(yy))
#
#        plt.quiver(xx, yy,
#                   u_arr,
#                   v_arr,
#                   zorder=2, color='magenta',
#                   scale_units='xy',
#                   width=0.005,
#                   scale=0.2e-6,
##                   width=0.5, 
#                   transform=ccrs.PlateCarree())
#
#        plt.show()


#    def test_values(self):
#        # Construct a rotated-pole grid and check angle calculation.
#        testcube = _rotated_grid_sample()
#
#        cell_angle_boundpoints = 'mid-lhs, mid-rhs'
##        cell_angle_boundpoints = 'lower-left, lower-right'
##        cell_angle_boundpoints = 'garble'
#        angles_cube = gridcell_angles(
#            testcube,
#            cell_angle_boundpoints=cell_angle_boundpoints)
#        angles_cube.convert_units('radians')
#
#        # testing phase...
#        print(np.rad2deg(angles_cube.data))
#
#        import matplotlib.pyplot as plt
#        plt.switch_backend('tkagg')
#
##        plot_map = 'north_polar_stereographic'
##        plot_map = 'plate_carree'
##        plot_map = 'mercator'
#        plot_map = 'north_polar_orthographic'
#        if plot_map == 'plate_carree':
#            scale = 0.1
#            map_proj = ccrs.PlateCarree()
#        elif plot_map == 'mercator':
#            scale = 3.0e-6
#            map_proj = ccrs.Mercator()
#            map_proj._threshold *= 0.01
#        elif plot_map == 'north_polar_orthographic':
#            scale = 3.0e-6
#            map_proj = ccrs.Orthographic(central_longitude=0.0,
#                                         central_latitude=90.0,)
#            map_proj._threshold *= 0.01
#        elif plot_map == 'north_polar_stereographic':
#            scale = 3.0e-6
#            map_proj = ccrs.NorthPolarStereo()
#        else:
#            assert 0
#
#        ax = plt.axes(projection=map_proj)
#        data_proj = ccrs.PlateCarree()
#
#        deg_scale = 10.0
#
##        angles = 'uv'
#        angles = 'xy'
#
#        ax.coastlines()
#        ax.gridlines()
#        for i_bnd in range(4):
#            color = ['black', 'red', 'blue', 'magenta'][i_bnd]
#            plt.plot(testcube.coord('longitude').bounds[..., i_bnd],
#                     testcube.coord('latitude').bounds[..., i_bnd],
#                     '+', markersize=10., markeredgewidth=2.,
#                     markerfacecolor=color, markeredgecolor=color,
#                     transform=data_proj)
#
#
#        # Show plain 0,1 + 1,0 (PlateCarree) vectors unrotated at the given points.
#        pts_shape = testcube.coord('longitude').shape
#        ny, nx = pts_shape
#        u0 = np.ones(pts_shape)
#        v0 = np.zeros(pts_shape)
#        u1 = v0.copy()
#        v1 = u0.copy()
#
#        x0s = testcube.coord('longitude').points
#        y0s = testcube.coord('latitude').points
#        yscale = np.cos(np.deg2rad(y0s))
#        plt.quiver(x0s, y0s, u0, v0 * yscale,
#                   color='blue', width=0.005,
#                   headwidth=2., # headlength=1.0, headaxislength=0.7,
#                   angles=angles,
#                   scale_units='xy', scale=scale,
#                   transform=data_proj)
#        plt.quiver(x0s, y0s, u1, v1 * yscale,
#                   color='red', width=0.005,
#                   headwidth=2., # headlength=1.0, headaxislength=0.7,
#                   angles=angles,
#                   scale_units='xy', scale=scale,
#                   transform=data_proj)
#
#        # Add 45deg arrows (NEly), still on a PlateCarree map.
#        plt.quiver(x0s, y0s, v1, v1 * yscale,
#                   color='green',  width=0.005,
#                   headwidth=2., # headlength=1.0, headaxislength=0.7,
#                   angles=angles,
#                   scale_units='xy', scale=scale,
#                   transform=data_proj)
#
#
#
#        #
#        # Repeat the above plotting short lines INSTEAD of quiver.
#        #
#        u0d = x0s + deg_scale * u0
#        v0d = y0s + deg_scale * v0
#        u1d = x0s + deg_scale * u1
#        v1d = y0s + deg_scale * v1
#        u2d = x0s + deg_scale * u0
#        v2d = y0s + deg_scale * v1
#        for iy in range(ny):
#            for ix in range(nx):
#                plt.plot([x0s[iy, ix], u0d[iy, ix]],
#                         [y0s[iy, ix], v0d[iy, ix]],
#                         ':', color='blue', linewidth=0.5,
#                         transform=data_proj)
#                plt.plot([x0s[iy, ix], u1d[iy, ix]],
#                         [y0s[iy, ix], v1d[iy, ix]],
#                         ':', color='red', linewidth=0.5,
#                         transform=data_proj)
#                plt.plot([x0s[iy, ix], u2d[iy, ix]],
#                         [y0s[iy, ix], v2d[iy, ix]],
#                         ':', color='green', linewidth=0.5,
#                         transform=data_proj)
#
#
#        # Overplot BL-BR and BL-TL lines from the cell bounds.
#        co_lon, co_lat = [testcube.coord(name).copy()
#                          for name in ('longitude', 'latitude')]
#        for co in (co_lon, co_lat):
#            co.convert_units('degrees')
#        lon_bds, lat_bds = [co.bounds for co in (co_lon, co_lat)]
##        ny, nx = lon_bds.shape[:-1]
#        for iy in range(ny):
#            for ix in range(nx):
#                x0, y0 = lon_bds[iy, ix, 0], lat_bds[iy, ix, 0]
#                x1, y1 = lon_bds[iy, ix, 1], lat_bds[iy, ix, 1]
#                x2, y2 = lon_bds[iy, ix, 3], lat_bds[iy, ix, 3]
#                plt.plot([x0, x1], [y0, y1], 'x-', 
#                         color='orange',
#                         transform=data_proj)
#                plt.plot([x0, x2], [y0, y2], 'x-', 
#                         color='orange', linestyle='--',
#                         transform=data_proj)
#
#        # Plot U0, rotated by cell angles, also at cell bottom-lefts.
#        u0_cube, u1_cube, v0_cube, v1_cube = [testcube.copy(data=aa)
#                                              for aa in (u0, v0, u1, v1)]
#        u0r_cube, v0r_cube = rotate_grid_vectors(
#            u0_cube, v0_cube, grid_angles_cube=angles_cube)
#        u0r, v0r = [cube.data for cube in (u0r_cube, v0r_cube)]
#
#        xbl, ybl = lon_bds[..., 0], lat_bds[..., 0]
#        #
#        # Replace quiver here with delta-based lineplot
#        #
#        urd = xbl + deg_scale * u0r
#        vrd = ybl + deg_scale * v0r * yscale
#        for iy in range(ny):
#            for ix in range(nx):
#                plt.plot([xbl[iy, ix], urd[iy, ix]],
#                         [ybl[iy, ix], vrd[iy, ix]],
#                         ':', color='magenta', linewidth=2.5,
#                         transform=data_proj)
#        # Show this is the SAME as lineplot
#        plt.quiver(xbl, ybl, u0r, v0r * yscale,
#                   color='magenta', width=0.01,
#                   headwidth=1.2, # headlength=1.0, headaxislength=0.7,
#                   angles=angles,
#                   scale_units='xy', scale=scale,
#                   transform=data_proj)
#
#        plt.suptitle('angles from "{}"'.format(cell_angle_boundpoints))
#
##        # Also draw small lines pointing at the correct (TRUE, not ) angle.
##        ny, nx = x0s.shape
##        size_degrees = 1.0
##        angles = angles_cube.copy()
##        angles.convert_units('radians')
##        angles = angles.data
##        lats = testcube.coord('latitude').copy()
##        lats.convert_units('radians')
##        lats = lats.points
##        dxs = size_degrees * u0.copy()  #* np.cos(angles)
##        dys = size_degrees * u0.copy()  # / np.sqrt(np.cos(lats))
##        x1s = x0s + dxs
##        y1s = y0s + dys
###        for iy in range(ny):
###            for ix in range(nx):
###                plt.plot([x0s[iy, ix], x1s[iy, ix]],
###                         [y0s[iy, ix], y1s[iy, ix]],
###                         'o-', markersize=4., markeredgewidth=0.,
###                         color='green', # scale_units='xy', scale=scale,
###                         transform=data_proj)
##        plt.quiver(x0s, y0s, dxs, dys,
##                 color='green', linewidth=0.2,
##                 angles=angles,
##                 scale_units='xy', scale=scale * 0.6,
##                 transform=data_proj)
#
#
#
#        ax.set_global()
#        plt.show()
#
#        angles_cube.convert_units('degrees')
#
#        self.assertArrayAllClose(
#            angles_cube.data,
#            [[33.421, 17.928, 0., -17.928, -33.421],
#             [41.981, 24.069, 0., -24.069, -41.981],
#             [56.624, 37.809, 0., -37.809, -56.624],
#             [79.940, 74.227, 0., -74.227, -79.940],
#             [107.313, 126.361, -180., -126.361, -107.313]],
#            atol=0.002)


if __name__ == "__main__":
    tests.main()
