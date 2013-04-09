# (C) British Crown Copyright 2013 Met Office
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
Test the :func:`iris.experimental.regrid._get_xy_dim_coords` function.

"""
from __future__ import print_function

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import os
import sys

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import ESMF

import cartopy.crs as ccrs
import iris
import iris.analysis
import iris.analysis.cartography as i_cartog
import iris.plot as iplt
import iris.quickplot as qplt
import iris.tests.stock as istk

from iris.experimental.regrid_conservative import regrid_conservative_via_esmpy


_debug = False
#_debug = True
if _debug:
    def dprint(*args):
        print( *args)
else:
    def dprint(*args):
        pass

def _make_test_cube(shape, xlims, ylims, pole_latlon=None):
    """ Create latlon cube (optionally rotated) with given xy dims+lims. """
    nx, ny = shape
    cube = iris.cube.Cube(np.zeros((ny, nx)))
    xvals = np.linspace(xlims[0], xlims[1], nx)
    yvals = np.linspace(ylims[0], ylims[1], ny)
    coordname_prefix = ''
    cs = iris.coord_systems.GeogCS(i_cartog.DEFAULT_SPHERICAL_EARTH_RADIUS)
    if pole_latlon is not None:
        coordname_prefix = 'grid_'
        pole_lat, pole_lon = pole_latlon
        cs = iris.coord_systems.RotatedGeogCS(
            grid_north_pole_latitude=pole_lat,
            grid_north_pole_longitude=pole_lon,
            ellipsoid=cs)

    co_x = iris.coords.DimCoord(xvals,
                                standard_name=coordname_prefix + 'longitude',
                                units=iris.unit.Unit('degrees'),
                                coord_system=cs)
    co_x.guess_bounds()
    cube.add_dim_coord(co_x, 1)
    co_y = iris.coords.DimCoord(yvals,
                                standard_name=coordname_prefix + 'latitude',
                                units=iris.unit.Unit('degrees'),
                                coord_system=cs)
    co_y.guess_bounds()
    cube.add_dim_coord(co_y, 0)
    return cube


def _cube_area_sum(cube):
    """ Calculate total area-sum, as Iris can not do in one operation. """
    area_sums = cube * i_cartog.area_weights(cube, normalize=False)
    return area_sums.collapsed(area_sums.coords(dim_coords=True), iris.analysis.SUM).data[0]


class TestConservativeRegrid(tests.IrisTest):
    @classmethod
    def setUpClass(self):
        # Pre-initialise ESMF, just to avoid warnings about no logfile.
        # NOTE: noisy if logging off, and no control of filepath.  Boo !!
        self._emsf_logfile_path = os.path.join(os.getcwd(), 'ESMF_LogFile')
        ESMF.Manager(logkind=ESMF.LogKind.SINGLE, debug=False)

    @classmethod
    def tearDownClass(self):
        # remove the logfile if we can, just to be tidy
        if os.path.exists(self._emsf_logfile_path):
            os.remove(self._emsf_logfile_path)

    def test_simple_areas(self):
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0
        c1_sum = np.sum(c1.data)
        c1_areasum = _cube_area_sum(c1)
        dprint('simple: c1 area-sum=', c1_areasum, ' cells-sum=', c1_sum)

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0

        # main regrid
        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        c1to2_sum = np.sum(c1to2.data)
        c1to2_areasum = _cube_area_sum(c1to2)
        dprint('simple: c1to2 area-sum=', c1to2_areasum, ' cells-sum=', c1to2_sum)

        d_expect = np.array([[0.00, 0.00, 0.00, 0.00],
                             [0.00, 0.25, 0.25, 0.00],
                             [0.00, 0.25, 0.25, 0.00],
                             [0.00, 0.00, 0.00, 0.00]])
        # Numbers are slightly off (~0.25000952).  This is expected.
        self.assertArrayAllClose(c1to2.data, d_expect, rtol=5.0e-5)

        # check that the area sums are equivalent, simple total is a bit off
        self.assertAlmostEqual(c1to2_sum, 1.0, delta=0.00005)
        self.assertArrayAllClose(c1to2_areasum, c1_areasum)

        # regrid back onto original grid again
        c1to2to1 = regrid_conservative_via_esmpy(c1to2, c1)

        c1to2to1_sum = np.sum(c1to2to1.data)
        c1to2to1_areasum = _cube_area_sum(c1to2to1)
        dprint('simple: c1to2to1 area-sum=', c1to2to1_areasum, ' cells-sum=', c1to2to1_sum)

        d_expect = np.array([[0.0, 0.0000, 0.0000, 0.0000, 0.0],
                             [0.0, 0.0625, 0.1250, 0.0625, 0.0],
                             [0.0, 0.1250, 0.2500, 0.1250, 0.0],
                             [0.0, 0.0625, 0.1250, 0.0625, 0.0],
                             [0.0, 0.0000, 0.0000, 0.0000, 0.0]])
        # Errors now quite large
        self.assertArrayAllClose(c1to2to1.data, d_expect, atol=0.00002)

        # check area sums again
        self.assertAlmostEqual(c1to2_sum, 1.0, delta=0.00008)
        self.assertArrayAllClose(c1to2to1_areasum, c1_areasum)

    def test_polar_areas(self):
        # Like test_basic_area, but not symmetrical + bigger overall errors.
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (84, 88))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0
        c1_sum = np.sum(c1.data)
        c1_areasum = _cube_area_sum(c1)
        dprint('polar: c1 area-sum=', c1_areasum, ' cells-sum=', c1_sum)

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (84.5, 87.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0

        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        c1to2_sum = np.sum(c1to2.data)
        c1to2_areasum = _cube_area_sum(c1to2)
        dprint('polar: c1to2 area-sum=', c1to2_areasum, ' cells-sum=', c1to2_sum)

        # check for expected pattern
        d_expect = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.23614, 0.23614, 0.0],
                             [0.0, 0.26784, 0.26784, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])
        self.assertArrayAllClose(c1to2.data, d_expect, rtol=5.0e-5)

        # check sums
        self.assertAlmostEqual(c1to2_sum, 1.0, delta=0.008)
        self.assertArrayAllClose(c1to2_areasum, c1_areasum)

        c1to2to1 = regrid_conservative_via_esmpy(c1to2, c1)
        c1to2to1_sum = np.sum(c1to2to1.data)
        c1to2to1_areasum = _cube_area_sum(c1to2to1)
        dprint('polar: c1to2to1 area-sum=', c1to2to1_areasum, ' cells-sum=', c1to2to1_sum)

        d_expect = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.056091, 0.112181, 0.056091, 0.0],
                             [0.0, 0.125499, 0.250998, 0.125499, 0.0],
                             [0.0, 0.072534, 0.145067, 0.072534, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertArrayAllClose(c1to2to1.data, d_expect, atol=0.0005)

        # check sums
        self.assertAlmostEqual(c1to2to1_sum, 1.0, delta=0.02)
        self.assertArrayAllClose(c1to2to1_areasum, c1_areasum)

    def test_fail_no_cs(self):
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0
        c2.coord('latitude').coord_system = None

        with self.assertRaises(ValueError):
            c1to2 = regrid_conservative_via_esmpy(c1, c2)

    def test_fail_different_cs(self):
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        c1 = _make_test_cube(shape1, xlims1, ylims1,
                             pole_latlon=(45.0, 35.0))
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0
        c2.coord('latitude').coord_system = \
            c1.coord('grid_latitude').coord_system

        with self.assertRaises(ValueError):
            c1to2 = regrid_conservative_via_esmpy(c1, c2)

    def test_rotated(self):
        """
        Perform area-weighted regrid on more complex area.

        Use two mutually rotated grids, of similar area + same dims.
        Use a small central region in each which is entirely within the other region.
        """
        # create source test cube on rotated form
        pole_lat = 53.4
        pole_lon = -173.2
        deg_swing = 35.3
        pole_lon += deg_swing
        c1_nx = 9 + 6
        c1_ny = 7 + 6
        c1_xlims = -60.0, 60.0
        c1_ylims = -45.0, 20.0
        c1_xlims = [x - deg_swing for x in c1_xlims]
        c1 = _make_test_cube((c1_nx, c1_ny), c1_xlims, c1_ylims,
                             pole_latlon=(pole_lat, pole_lon))
        c1.data[3:-3, 3:-3] = np.array([
            [100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 199, 199, 199, 199, 100, 100, 100],
            [100, 100, 100, 100, 199, 199, 100, 100, 100],
            [100, 100, 100, 100, 199, 199, 199, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100]],
            dtype=np.float)

        c1_sum = np.sum(c1.data)
        c1_areasum = _cube_area_sum(c1)
        dprint('rotate: c1 area-sum=', c1_areasum, ' cells-sum=', c1_sum)

        # construct target cube to receive
        nx2 = 9 + 6
        ny2 = 7 + 6
        c2_xlims = -100.0, 120.0
        c2_ylims = -20.0, 50.0
        c2 = _make_test_cube((nx2, ny2), c2_xlims, c2_ylims)

        # perform regrid + snapshot test results
        c1to2 = regrid_conservative_via_esmpy(c1, c2)
        # just check we have zeros all around the edge..
        self.assertArrayEqual(c1to2.data[0, :], 0.0)
        self.assertArrayEqual(c1to2.data[-1, :], 0.0)
        self.assertArrayEqual(c1to2.data[:, 0], 0.0)
        self.assertArrayEqual(c1to2.data[:, -1], 0.0)
        # check the area-sum operation
        c1to2_sum = np.sum(c1to2.data)
        c1to2_areasum = _cube_area_sum(c1to2)
        dprint('rotate: c1to2 area-sum=', c1to2_areasum, ' cells-sum=', c1to2_sum)

        def reldiff(a, b):
            if a == 0.0 and b == 0.0:
                return 0.0
            return abs(a-b)*2.0/(abs(a)+abs(b))

        dprint('rotate: c1to2/c1 area-sum rel-diff = ', reldiff(c1to2_areasum, c1_areasum))
        self.assertArrayAllClose(c1to2_areasum, c1_areasum, rtol=0.004)

#        levels = [1.0, 50, 99, 120, 140, 160, 180]
#
#        plt.figure()
#        ax = plt.axes(projection=ccrs.PlateCarree())
#        iplt.contourf(c2, levels=levels, extend='both')
#        p1 = iplt.contourf(c1, levels=levels, extend='both')
#        plt.colorbar(p1)
#        ax.xaxis.set_visible(True)
#        ax.yaxis.set_visible(True)
#        plt.show(block=False)
#
#        plt.figure()
#        ax = plt.axes(projection=ccrs.PlateCarree())
#        iplt.contourf(c2, levels=levels, extend='both')
#        p2 = iplt.pcolormesh(c1to2)
#        plt.colorbar(p2)
#        ax.xaxis.set_visible(True)
#        ax.yaxis.set_visible(True)
#        plt.show()


        #
        # Now repeat transform backwards ...
        #
        c2.data[5:-5, 5:-5] = np.array([
            [199, 199, 199, 199, 100],
            [100, 100, 199, 199, 100],
            [100, 100, 199, 199, 199]],
            dtype=np.float)
        c2_sum = np.sum(c2.data)
        c2_areasum = _cube_area_sum(c2)
        dprint('back-rotate: c2 area-sum=', c2_areasum, ' cells-sum=', c2_sum)

        c2to1 = regrid_conservative_via_esmpy(c2, c1)
        # just check we have zeros all around the edge..
        self.assertArrayEqual(c2to1.data[0, :], 0.0)
        self.assertArrayEqual(c2to1.data[-1, :], 0.0)
        self.assertArrayEqual(c2to1.data[:, 0], 0.0)
        self.assertArrayEqual(c2to1.data[:, -1], 0.0)

        c2to1_sum = np.sum(c2to1.data)
        c2to1_areasum = _cube_area_sum(c2to1)
        dprint('rotate: c2to1 area-sum=', c2to1_areasum, ' cells-sum=', c2to1_sum)

        dprint('rotate: c2to1/c2 area-sum rel-diff = ', reldiff(c2to1_areasum, c2_areasum))
        self.assertArrayAllClose(c2to1_areasum, c2_areasum, rtol=0.004)

#        levels = [1.0, 50, 99, 120, 140, 160, 180]
#
#        plt.figure()
##        ax = plt.axes(projection=ccrs.PlateCarree())
#        iplt.contourf(c1, levels=levels, extend='both')
#        p1 = iplt.contourf(c2, levels=levels, extend='both')
#        plt.colorbar(p1)
#        ax.xaxis.set_visible(True)
#        ax.yaxis.set_visible(True)
#        plt.show(block=False)
#
#        plt.figure()
##        ax = plt.axes(projection=ccrs.PlateCarree())
#        iplt.contourf(c1, levels=levels, extend='both')
#        p2 = iplt.pcolormesh(c2to1)
#        plt.colorbar(p2)
#        ax.xaxis.set_visible(True)
#        ax.yaxis.set_visible(True)
#        plt.show()

#    TODOs
#    ==== 20130408 ====
#    Testing areas to consider
#      * longitude wrapping
#      * global areas
#      * 1x1 cells
#      * ??1x1 global cell??
#      * MDI handling
#      * area checking
#        - (1) simple quasi-rectangular to show back + forth
#        - (2) near-pole stuff, proving true-area calc
#      * irregular grids (?)
#      * ? irregular bounds ?
#      * iterate over extra dimensions
#      * test with extra coords...
#        * additional (irrelevant) coords (DIM + other)
#        * factories aka hybrid-height
#
#      * data in regions that goes outside target area (MDI?)
#
#    TODO: area checks are wrong : need to use area-weighted sums...
#    TODO: masking !!
#    TODO: fix more general regional test with older tuned test-example
#            = _generate_test_cubes ...


if __name__ == '__main__':
    tests.main()
