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

import contextlib
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
_debug = True
_debug_pictures = False
#_debug_pictures = True

_debug_pictures &= _debug


def dprint(*args):
    if _debug:
        print(*args)


_plain_geodetic_cs = iris.coord_systems.GeogCS(
    i_cartog.DEFAULT_SPHERICAL_EARTH_RADIUS)

def _make_test_cube(shape, xlims, ylims, pole_latlon=None):
    """
    Create latlon cube (optionally rotated) with given xy dims+lims.

    Does not work for 1xN or Nx1 grids, because guess_bounds fails.
    """
    nx, ny = shape
    cube = iris.cube.Cube(np.zeros((ny, nx)))
    xvals = np.linspace(xlims[0], xlims[1], nx)
    yvals = np.linspace(ylims[0], ylims[1], ny)
    coordname_prefix = ''
    cs = _plain_geodetic_cs
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
    """ Calculate total area-sum - Iris can't do this in one operation. """
    area_sums = cube * i_cartog.area_weights(cube, normalize=False)
    area_sum = area_sums.collapsed(area_sums.coords(dim_coords=True),
                                   iris.analysis.SUM)
    return area_sum.data.flatten()[0]


def _reldiff(a, b):
    """
    Compute relative-difference measure between real numbers.

    Result is:
        if a == b == 0:
            0.0
        otherwise:
            |a - b| / mean(|a|, |b|)

    """
    if a == 0.0 and b == 0.0:
        return 0.0
    return abs(a - b) * 2.0 / (abs(a) + abs(b))


def _minmax(v):
    """ Calculate [min, max] of input. """
    return [f(v) for f in (np.min, np.max)]


class TestConservativeRegrid(tests.IrisTest):
    @classmethod
    def setUpClass(self):
        # Pre-initialise ESMF, just to avoid warnings about no logfile.
        # NOTE: noisy if logging is off, and no control of filepath.  Boo!!
        self._emsf_logfile_path = os.path.join(os.getcwd(), 'ESMF_LogFile')
        ESMF.Manager(logkind=ESMF.LogKind.SINGLE, debug=False)

    @classmethod
    def tearDownClass(self):
        # remove the logfile if we can, just to be tidy
        if not _debug:
            if os.path.exists(self._emsf_logfile_path):
                os.remove(self._emsf_logfile_path)

    def setUp(self):
        if _debug:
            # emit an extra linefeed for debug output (-v output omits one)
            dprint()
            # tweak array printouts
            np.set_printoptions(precision=2, linewidth=200, suppress=True)

        # Compute basic test data cubes.
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0

        # Save timesaving pre-computed bits
        self.stock_c1_c2 = (c1, c2)
        self.stock_regrid_c1toc2 = regrid_conservative_via_esmpy(c1, c2)
        self.stock_c1_areasum = _cube_area_sum(c1)

    def test_simple_areas(self):
        """
        Test area-conserving regrid between simple "near-square" grids.

        Grids have overlapping areas in the same (lat-lon) coordinate system.
        Grids are "nearly flat" lat-lon spaces (small ranges near the equator).

        """
        c1, c2 = self.stock_c1_c2
        c1_areasum = self.stock_c1_areasum

        # main regrid
        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        c1to2_areasum = _cube_area_sum(c1to2)

        # check all expected values
        d_expect = np.array([[0.00, 0.00, 0.00, 0.00],
                             [0.00, 0.25, 0.25, 0.00],
                             [0.00, 0.25, 0.25, 0.00],
                             [0.00, 0.00, 0.00, 0.00]])
        # Numbers are slightly off (~0.25000952).  This is expected.
        self.assertArrayAllClose(c1to2.data, d_expect, rtol=5.0e-5)

        # check that the area sums are equivalent, simple total is a bit off
        dprint('simple: c1 area-sum=', c1_areasum)
        dprint('simple: c1to2 area-sum=', c1to2_areasum)
        dprint('simple: REL-DIFF c1to2/c1 area-sum = ',
               _reldiff(c1to2_areasum, c1_areasum))
        self.assertArrayAllClose(c1to2_areasum, c1_areasum)

        #
        # regrid back onto original grid again ...
        #
        c1to2to1 = regrid_conservative_via_esmpy(c1to2, c1)

        c1to2to1_areasum = _cube_area_sum(c1to2to1)

        d_expect = np.array([[0.0, 0.0000, 0.0000, 0.0000, 0.0],
                             [0.0, 0.0625, 0.1250, 0.0625, 0.0],
                             [0.0, 0.1250, 0.2500, 0.1250, 0.0],
                             [0.0, 0.0625, 0.1250, 0.0625, 0.0],
                             [0.0, 0.0000, 0.0000, 0.0000, 0.0]])
        # Errors now quite large
        self.assertArrayAllClose(c1to2to1.data, d_expect, atol=0.00002)

        # check area sums again
        dprint('simple: c1to2to1 area-sum=', c1to2to1_areasum)
        dprint('simple: REL-DIFF c1to2to1/c1 area-sum = ',
               _reldiff(c1to2to1_areasum, c1_areasum))
        self.assertArrayAllClose(c1to2to1_areasum, c1_areasum)

    def test_xy_transposed(self):
        """ Test effects of transposing X and Y in src/dst data. """
        c1, c2 = self.stock_c1_c2
        testcube_xy = self.stock_regrid_c1toc2

        # Check that transposed data produces transposed results
        # - i.e.  regrid(data^T)^T == regrid(data)
        c1_yx = c1.copy()
        c1_yx.transpose()
        testcube_yx = regrid_conservative_via_esmpy(c1_yx, c2)
        testcube_yx.transpose()
        self.assertTrue( testcube_yx == testcube_xy )

        # Check that transposing destination does nothing
        c2_yx = c2.copy()
        c2_yx.transpose()
        testcube_dst_transpose = regrid_conservative_via_esmpy(c1, c2_yx)
        self.assertTrue( testcube_dst_transpose == testcube_xy )

    def test_dst_coords(self):
        """ Test specifying destination grid with coords instead of cube. """
        # Check that doing op with XY coords as dst is equivalent to the cube.
        c1, c2 = self.stock_c1_c2
        testcube = self.stock_regrid_c1toc2

        dst_x = c2.coord(axis='x')
        dst_y = c2.coord(axis='y')
        testcube_coords = regrid_conservative_via_esmpy(c1, (dst_x, dst_y))
        self.assertTrue( testcube_coords == testcube )

        # Check that swapping dst_coords produces transposed results.
        testcube_coords_yx = regrid_conservative_via_esmpy(c1, (dst_y, dst_x))
        testcube_coords_yx.transpose()
        self.assertTrue( testcube_coords_yx == testcube )

    def test_same_grid(self):
        """ Test regridding onto same grid. """
        # Check that doing op with self as target is equivalent to the original.
        c1, c2 = self.stock_c1_c2
        testcube = regrid_conservative_via_esmpy(c1, c1)
        self.assertTrue( testcube == c1 )
        self.assertArrayEqual(testcube.data, c1.data )

    def test_global(self):
        """ Test global regridding. """
        # Compute basic test data cubes.
        shape1 = (8, 6)
        xlim1 = 180.0 * (shape1[0] - 1) / shape1[0]
        ylim1 = 90.0 * (shape1[1] - 1) / shape1[1]
        c1 = _make_test_cube(shape1, (-xlim1, xlim1), (-ylim1, ylim1))
        # Create a small, plausible global array:
        # - top + bottom rows all the same
        # - left + right columns the same (though "close" would do)
        basedata = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1,],
             [1, 1, 4, 4, 4, 2, 2, 1,],
             [2, 1, 4, 4, 4, 2, 2, 2,],
             [2, 5, 5, 1, 1, 1, 5, 5],
             [5, 5, 5, 1, 1, 1, 5, 5],
             [5, 5, 5, 5, 5, 5, 5, 5]])
        c1.data[:] = basedata

        # Create a rotated grid to regrid this onto.
        shape2 = (14, 11)
        xlim2 = 180.0 * (shape2[0] - 1) / shape2[0]
        ylim2 = 90.0 * (shape2[1] - 1) / shape2[1]
        c2 = _make_test_cube(shape2, (-xlim2, xlim2), (-ylim2, ylim2),
                             pole_latlon=(47.4, 25.7))

        # Perform regridding
        c1toc2 = regrid_conservative_via_esmpy(c1, c2)

        if _debug_pictures:
            levels = np.linspace(0.0, np.max(basedata), 20)
            vmin, vmax = 0, np.max(basedata)

            plt.figure()
            c1_proj = c1.coord(axis='x').coord_system
            c1_proj = c1_proj.as_cartopy_projection()
            ax = plt.axes(projection=c1_proj)
            p1 = iplt.pcolormesh(c1, vmin=vmin, vmax=vmax, cmap='Blues')
            iplt.outline(c1)
            plt.colorbar(p1)
            plt.title('source')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            plt.figure()
            c2_proj = c2.coord(axis='x').coord_system
            c2_proj = c2_proj.as_cartopy_projection()
            ax = plt.axes(projection=c2_proj)
            p1 = iplt.pcolormesh(c1toc2, vmin=vmin, vmax=vmax, cmap='Blues',
                                 edgecolor='k')
            iplt.outline(c1toc2)
            iplt.outline(c1)
            plt.colorbar(p1)
            plt.title('dest with both outlines')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            plt.figure()
            ax = plt.axes(projection=c1_proj)
            p1 = iplt.pcolormesh(c1toc2, vmin=vmin, vmax=vmax, cmap='Blues')
            iplt.outline(c1toc2)
            plt.colorbar(p1)
            plt.title('dest on source projection')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=True)

        # Check that before+after area-sums match fairly well
        c1_areasum = _cube_area_sum(c1)
        c1toc2_areasum = _cube_area_sum(c1toc2)
        dprint('global: area-sums RELDIFF c1/c1toc2 = ',
               _reldiff(c1_areasum, c1toc2_areasum))
        self.assertArrayAllClose(c1toc2_areasum, c1_areasum, rtol=0.006)

    def test_global_collapse(self):
        # Fetch 'standard' testcube data
        c1, _ = self.stock_c1_c2
        c1_areasum = self.stock_c1_areasum

        # Condense entire globe onto a single cell
        shape2 = (1, 1)
        x_coord_2 = iris.coords.DimCoord([0.0], bounds=[-180.0, 180.0],
                                         coord_system=_plain_geodetic_cs)
        y_coord_2 = iris.coords.DimCoord([0.0], bounds=[-90.0, 90.0],
                                         coord_system=_plain_geodetic_cs)

        global_cell_supported = False
        # NOTE: at present, this causes an error inside ESMF ...
        if global_cell_supported:
            @contextlib.contextmanager
            def context():
                yield
        else:
            context = self.assertRaises(NameError)

        with context:
            c1_to_global = regrid_conservative_via_esmpy(c1,
                (x_coord_2, y_coord_2))

            # Check the total area sum is still the same
            dprint('global: area-sums RELDIFF orig/global = ',
                   _reldiff(c1_to_global.data[0,0], c1_areasum))
            self.assertArrayAllClose(c1_to_global.data[0,0], c1_areasum)

    def test_single_cells(self):
        # Fetch 'standard' testcube data
        c1, c2 = self.stock_c1_c2
        c1_areasum = self.stock_c1_areasum

#
# At present NxN -> 1x1 doesn't seem to work
#   - always gets misssing-data in cell ?
#
#
#        # Condense entire region into a single cell in the c1 grid
#        xlims1 = _minmax(c1.coord(axis='x').bounds)
#        ylims1 = _minmax(c1.coord(axis='y').bounds)
#        x_c1x1 = iris.coords.DimCoord(xlims1[0], bounds=xlims1,
#                                      long_name='longitude',
#                                      coord_system=_plain_geodetic_cs)
#        y_c1x1 = iris.coords.DimCoord(ylims1[0], bounds=ylims1,
#                                      long_name='latitude',
#                                      coord_system=_plain_geodetic_cs)
#        c1x1 = regrid_conservative_via_esmpy(c1, (x_c1x1, y_c1x1))
#
#        # Check the total area sum is still the same
#        c1x1_areasum = _cube_area_sum(c1x1)
#        dprint('single : area-sums RELDIFF NxN -> 1x1 = ',
#               _reldiff(c1x1_areasum, c1_areasum))
#        self.assertArrayAllClose(c1x1_areasum, c1_areasum)

        # Check reverse calculation back to c1 (i.e. *source* is 1x1)

        # construct an approximation of a collapsed cube with same area sum.
        # NOTE: can't use _make_cube (see docstring)
        c1x1 = c1.copy()[0:1,0:1]
        xlims1 = _minmax(c1.coord(axis='x').bounds)
        ylims1 = _minmax(c1.coord(axis='y').bounds)
        c1x1.coord(axis='x').bounds = xlims1
        c1x1.coord(axis='y').bounds = ylims1
        c1x1.data[0,0] = np.mean(c1.data)  #NOTE: not quite right, but should do

        # Regrid this back onto the original NxN grid
        c1x1_to_c1 = regrid_conservative_via_esmpy(c1x1, c1)
        c1x1_to_c1_areasum = _cube_area_sum(c1x1_to_c1)

        # Check that area sum is ~unchanged, as expected
        dprint('single : area-sums RELDIFF 1x1 -> NxN = ',
               _reldiff(c1x1_to_c1_areasum, c1_areasum))
        self.assertArrayAllClose(c1x1_to_c1_areasum, c1_areasum, 0.0004)

        # Finally, check 1x1 -> 1x1
        # NOTE: can only get any result with a fully overlapping cell, so just
        # use regrid-to-self
        c1x1toself = regrid_conservative_via_esmpy(c1x1, c1x1)
        c1x1toself_areasum = _cube_area_sum(c1x1toself)
        dprint('single : area-sums RELDIFF 1x1 -> 1x1 = ',
               _reldiff(c1x1toself_areasum, c1_areasum))
        self.assertArrayAllClose(c1x1toself_areasum, c1_areasum, 0.0004)

    def test_longitude_wraps(self):
        """ Check results are independent of where the grid 'seams' are. """
        # First repeat global regrid calculation from 'test_global'.
        shape1 = (8, 6)
        xlim1 = 180.0 * (shape1[0] - 1) / shape1[0]
        ylim1 = 90.0 * (shape1[1] - 1) / shape1[1]
        c1 = _make_test_cube(shape1, (-xlim1, xlim1), (-ylim1, ylim1))

        basedata = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1,],
             [1, 1, 4, 4, 4, 2, 2, 1,],
             [2, 1, 4, 4, 4, 2, 2, 2,],
             [2, 5, 5, 1, 1, 1, 5, 5],
             [5, 5, 5, 1, 1, 1, 5, 5],
             [5, 5, 5, 5, 5, 5, 5, 5]])
        c1.data[:] = basedata

        shape2 = (14, 11)
        xlim2 = 180.0 * (shape2[0] - 1) / shape2[0]
        ylim2 = 90.0 * (shape2[1] - 1) / shape2[1]
        xlims_2 = (-xlim2, xlim2)
        ylims_2 = (-ylim2, ylim2)
        c2 = _make_test_cube(shape2, xlims_2, ylims_2,
                             pole_latlon=(47.4, 25.7))

        # Perform regridding
        c1toc2 = regrid_conservative_via_esmpy(c1, c2)

        if _debug_pictures:
            levels = np.linspace(0.0, np.max(basedata), 20)
            vmin, vmax = 0, np.max(basedata)

            plt.figure()
            c1_proj = c1.coord(axis='x').coord_system
            c1_proj = c1_proj.as_cartopy_projection()
            ax = plt.axes(projection=c1_proj)
            p1 = iplt.pcolormesh(c1, vmin=vmin, vmax=vmax, cmap='Blues')
            iplt.outline(c1)
            plt.colorbar(p1)
            plt.title('source')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            plt.figure()
#            c2_proj = c2.coord(axis='x').coord_system
#            c2_proj = c2_proj.as_cartopy_projection()
#            ax = plt.axes(projection=c2_proj)
            p1 = plt.pcolormesh(c1toc2.data, vmin=vmin, vmax=vmax, cmap='Blues',
                                 edgecolor='k')
#            iplt.outline(c1toc2)
#            iplt.outline(c1)
            plt.colorbar(p1)
            plt.title('dest')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            plt.figure()
            ax = plt.axes(projection=c1_proj)
            p1 = iplt.pcolormesh(c1toc2, vmin=vmin, vmax=vmax, cmap='Blues')
            iplt.outline(c1toc2)
            plt.colorbar(p1)
            plt.title('dest on source projection')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

        # Now Redo with dst longitudes rotated, so the 'seam' goes somewhere else.
        x_shift_steps = int(shape2[0] / 3)
        xlims2_shifted = np.array(xlims_2) + 360.0 * x_shift_steps / shape2[0]
        c2_shifted = _make_test_cube(shape2, xlims2_shifted, ylims_2,
                             pole_latlon=(47.4, 25.7))
        c1toc2_shifted = regrid_conservative_via_esmpy(c1, c2_shifted)

        if _debug_pictures:
            levels = np.linspace(0.0, np.max(basedata), 20)
            vmin, vmax = 0, np.max(basedata)

#            plt.figure()
            c1_proj = c1.coord(axis='x').coord_system
            c1_proj = c1_proj.as_cartopy_projection()
#            ax = plt.axes(projection=c1_proj)
#            p1 = iplt.pcolormesh(c1, vmin=vmin, vmax=vmax, cmap='Blues')
#            iplt.outline(c1)
#            plt.colorbar(p1)
#            ax.xaxis.set_visible(True)
#            ax.yaxis.set_visible(True)
#            plt.show(block=False)

            plt.figure()
#            c2_proj = c2_shifted.coord(axis='x').coord_system
#            c2_proj = c2_proj.as_cartopy_projection()
#            ax = plt.axes(projection=c2_proj)
            p1 = plt.pcolormesh(c1toc2_shifted.data,
                                vmin=vmin, vmax=vmax, cmap='Blues',
                                edgecolor='k')
#            iplt.outline(c1toc2_shifted)
#            iplt.outline(c1)
            plt.colorbar(p1)
            plt.title('dest_SHIFTED')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            plt.figure()
            ax = plt.axes(projection=c1_proj)
            p1 = iplt.pcolormesh(c1toc2_shifted, vmin=vmin, vmax=vmax, cmap='Blues')
            iplt.outline(c1toc2_shifted)
            plt.colorbar(p1)
            plt.title('dest_SHIFTED on source projection')
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=True)

        # Show that results are the same, when output rolled by same amount
        rolled_data = np.roll(c1toc2_shifted.data, x_shift_steps, axis=1)
        dprint('wraps: old+new maxdiff = ', np.max(np.abs(rolled_data - c1toc2.data)))
        self.assertArrayAllClose(rolled_data, c1toc2.data)

        # TODO: show that nothing changes when you rotate the SOURCE data

    def test_polar_areas(self):
        """
        Test area-conserving regrid between different grids.

        Grids have overlapping areas in the same (lat-lon) coordinate system.
        Cells are highly non-square (near the pole).

        """
        # Like test_basic_area, but not symmetrical + bigger overall errors.
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (84, 88))
        c1 = _make_test_cube(shape1, xlims1, ylims1)
        c1.data[:] = 0.0
        c1.data[2, 2] = 1.0
        c1_areasum = _cube_area_sum(c1)

        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (84.5, 87.5))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.data[:] = 0.0

        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        # check for expected pattern
        d_expect = np.array([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.23614, 0.23614, 0.0],
                             [0.0, 0.26784, 0.26784, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])
        self.assertArrayAllClose(c1to2.data, d_expect, rtol=5.0e-5)

        # check sums
        c1to2_areasum = _cube_area_sum(c1to2)
        dprint('polar: c1 area-sum=', c1_areasum)
        dprint('polar: c1to2 area-sum=', c1to2_areasum)
        dprint('polar: REL-DIFF c1to2/c1 area-sum = ',
               _reldiff(c1to2_areasum, c1_areasum))
        self.assertArrayAllClose(c1to2_areasum, c1_areasum)

        #
        # transform back again ...
        #
        c1to2to1 = regrid_conservative_via_esmpy(c1to2, c1)

        # check values
        d_expect = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.056091, 0.112181, 0.056091, 0.0],
                             [0.0, 0.125499, 0.250998, 0.125499, 0.0],
                             [0.0, 0.072534, 0.145067, 0.072534, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertArrayAllClose(c1to2to1.data, d_expect, atol=0.0005)

        # check sums
        c1to2to1_areasum = _cube_area_sum(c1to2to1)
        dprint('polar: c1to2to1 area-sum=', c1to2to1_areasum)
        dprint('polar: REL-DIFF c1to2to1/c1 area-sum = ',
               _reldiff(c1to2to1_areasum, c1_areasum))
        self.assertArrayAllClose(c1to2to1_areasum, c1_areasum)

    def test_fail_no_cs(self):
        """
        Check error if one coordinate has no coord_system.
        """
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
        """
        Check error when coordinates have different coord_systems.
        """
        shape1 = (5, 5)
        xlims1, ylims1 = ((-2, 2), (-2, 2))
        shape2 = (4, 4)
        xlims2, ylims2 = ((-1.5, 1.5), (-1.5, 1.5))

        # Check basic regrid between these is ok.
        c1 = _make_test_cube(shape1, xlims1, ylims1,
                             pole_latlon=(45.0, 35.0))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        regrid_conservative_via_esmpy(c1, c2)

        # Replace the coord_system one of the source coords + check this fails.
        c1.coord('grid_longitude').coord_system = \
            c2.coord('longitude').coord_system
        with self.assertRaises(ValueError):
            regrid_conservative_via_esmpy(c1, c2)

        # Repeat with target coordinate fiddled.
        c1 = _make_test_cube(shape1, xlims1, ylims1,
                             pole_latlon=(45.0, 35.0))
        c2 = _make_test_cube(shape2, xlims2, ylims2)
        c2.coord('latitude').coord_system = \
            c1.coord('grid_latitude').coord_system
        with self.assertRaises(ValueError):
            regrid_conservative_via_esmpy(c1, c2)

    def test_rotated(self):
        """
        Test area-weighted regrid on more complex area.

        Use two mutually rotated grids, of similar area + same dims.
        Only a small central region in each is non-zero, which maps entirely
        inside the other region.
        So the area-sum totals should match exactly.
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

        c1_areasum = _cube_area_sum(c1)
        dprint('rotate: c1 area-sum=', c1_areasum)

        # construct target cube to receive
        nx2 = 9 + 6
        ny2 = 7 + 6
        c2_xlims = -100.0, 120.0
        c2_ylims = -20.0, 50.0
        c2 = _make_test_cube((nx2, ny2), c2_xlims, c2_ylims)
        c2.data = np.ma.array(c2.data, mask=True)

        # perform regrid
        c1to2 = regrid_conservative_via_esmpy(c1, c2)

        # check we have zeros (or nearly) all around the edge..
        c1toc2_zeros = np.ma.array(c1to2.data)
        c1toc2_zeros[c1toc2_zeros.mask] = 0.0
        c1toc2_zeros = np.abs(c1toc2_zeros.mask) < 1.0e-6
        self.assertArrayEqual(c1toc2_zeros[0, :], True)
        self.assertArrayEqual(c1toc2_zeros[-1, :], True)
        self.assertArrayEqual(c1toc2_zeros[:, 0], True)
        self.assertArrayEqual(c1toc2_zeros[:, -1], True)

        # check the area-sum operation
        c1to2_areasum = _cube_area_sum(c1to2)
        dprint('rotate: c1to2 area-sum=', c1to2_areasum)
        dprint('rotate: REL-DIFF c1to2/c1 area-sum = ',
               _reldiff(c1to2_areasum, c1_areasum))
        self.assertArrayAllClose(c1to2_areasum, c1_areasum, rtol=0.004)

        if _debug_pictures:
            levels = [1.0, 50, 99, 120, 140, 160, 180]
            xlims, ylims = [_minmax(c2.coord(axis=ax).bounds)
                            for ax in ('x', 'y')]
            plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            iplt.contourf(c2, levels=levels, extend='both')
            p1 = iplt.contourf(c1, levels=levels, extend='both')
            iplt.outline(c2)
#            plt.colorbar(p1)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            iplt.contourf(c2, levels=levels, extend='both')
            p2 = iplt.pcolormesh(c1to2)
            iplt.outline(c2)
#            plt.colorbar(p2)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show()

        #
        # Now repeat, transforming backwards ...
        #
        c1.data = np.ma.array(c1.data, mask=True)
        c2.data[:] = 0.0
        c2.data[5:-5, 5:-5] = np.array([
            [199, 199, 199, 199, 100],
            [100, 100, 199, 199, 100],
            [100, 100, 199, 199, 199]],
            dtype=np.float)
        c2_areasum = _cube_area_sum(c2)
        dprint('back-rotate: c2 area-sum=', c2_areasum)

        c2toc1 = regrid_conservative_via_esmpy(c2, c1)

        # check we have zeros (or nearly) all around the edge..
        c2toc1_zeros = np.ma.array(c2toc1.data)
        c2toc1_zeros[c2toc1_zeros.mask] = 0.0
        c2toc1_zeros = np.abs(c2toc1_zeros.mask) < 1.0e-6
        self.assertArrayEqual(c2toc1_zeros[0, :], True)
        self.assertArrayEqual(c2toc1_zeros[-1, :], True)
        self.assertArrayEqual(c2toc1_zeros[:, 0], True)
        self.assertArrayEqual(c2toc1_zeros[:, -1], True)

        # check the area-sum operation
        c2toc1_areasum = _cube_area_sum(c2toc1)
        dprint('back-rotate: c2toc1 area-sum=', c2toc1_areasum)
        dprint('back-rotate: REL-DIFF c2toc1/c2 area-sum = ',
               _reldiff(c2toc1_areasum, c2_areasum))
        self.assertArrayAllClose(c2toc1_areasum, c2_areasum, rtol=0.004)

        if _debug_pictures:
            levels = [1.0, 50, 99, 120, 140, 160, 180]

            plt.figure()
            c2_proj = c2.coord(axis='x').coord_system
            c2_proj = c2_proj.as_cartopy_projection()
            ax = plt.axes(projection=c2_proj)
            p1 = iplt.contourf(c2, levels=levels, extend='both')
#            plt.colorbar(p1)
            ax.set_xlim()
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            xlims, ylims = [_minmax(c1.coord(axis=ax).bounds)
                            for ax in ('x', 'y')]
            plt.figure()
            ax = plt.axes()
            iplt.contourf(c1, levels=levels, extend='both')
            p1 = iplt.contourf(c2, levels=levels, extend='both')
            iplt.outline(c1)
#            plt.colorbar(p1)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show(block=False)

            plt.figure()
            iplt.contourf(c1, levels=levels, extend='both')
            p2 = iplt.pcolormesh(c2toc1)
            iplt.outline(c1)
#            plt.colorbar(p2)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.show()

    def test_missing_data_rotated(self):
        for do_add_missing in (False, True):
            debug_prefix = 'missing-data({}): '.format(
                'some' if do_add_missing else 'none')
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
            c1.data = np.ma.array(c1.data, mask=False)
            c1.data[3:-3, 3:-3] = np.ma.array([
                [100, 100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 199, 199, 199, 199, 100, 100, 100],
                [100, 100, 100, 100, 199, 199, 100, 100, 100],
                [100, 100, 100, 100, 199, 199, 199, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100, 100]],
                dtype=np.float)

            if do_add_missing:
                c1.data = np.ma.array(c1.data)
                c1.data[7, 7] = np.ma.masked
                c1.data[3:5, 10:12] = np.ma.masked

            # construct target cube to receive
            nx2 = 9 + 6
            ny2 = 7 + 6
            c2_xlims = -80.0, 80.0
            c2_ylims = -20.0, 50.0
            c2 = _make_test_cube((nx2, ny2), c2_xlims, c2_ylims)
            c2.data = np.ma.array(c2.data, mask=True)

            # perform regrid + snapshot test results
            c1toc2 = regrid_conservative_via_esmpy(c1, c2)

            if _debug_pictures:
                levels = [1.0, 50, 99, 120, 140, 160, 180]

                plt.figure()
                c1_proj = c1.coord(axis='x').coord_system
                c1_proj = c1_proj.as_cartopy_projection()
                ax = plt.axes(projection=c1_proj)
                p1 = iplt.contourf(c1, levels=levels, extend='both')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                plt.show(block=False)

                xlims, ylims = [_minmax(c2.coord(axis=ax).bounds)
                                for ax in ('x', 'y')]
                plt.figure()
                ax = plt.axes(projection=ccrs.PlateCarree())
                iplt.contourf(c2, levels=levels, extend='both')  # for axes box
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
#                p1 = iplt.contourf(c1, levels=levels, extend='both')
                iplt.pcolormesh(c1)
                iplt.outline(c2)
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                plt.show(block=False)

                plt.figure()
                ax = plt.axes(projection=ccrs.PlateCarree())
                iplt.contourf(c2, levels=levels, extend='both')
                iplt.outline(c2)
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                p2 = iplt.pcolormesh(c1toc2)
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                plt.show()

            # check masking of result is as expected
            # (generated by inspecting plot of how src+dst grids overlap)
            expected_mask_valuemap = np.array(
                # KEY: 0=masked, 7=present, 5=masked with masked datapoints
                [[0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
                 [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0],
                 [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 0, 0],
                 [0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 0, 0],
                 [0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 0, 0],
                 [0, 0, 0, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 0, 0],
                 [0, 0, 0, 0, 7, 7, 7, 5, 5, 7, 7, 7, 7, 0, 0],
                 [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0],
                 [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0],
                 [0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 0]])

            if do_add_missing:
                expected_mask = expected_mask_valuemap < 7
            else:
                expected_mask = expected_mask_valuemap == 0

            actual_mask = c1toc2.data.mask
            self.assertArrayEqual(actual_mask, expected_mask)

            if not do_add_missing:
                # check preservation of area-sums
                # NOTE: does *not* work with missing data, even theoretically,
                # as the 'missing areas' are not the same.
                c1_areasum = _cube_area_sum(c1)
                c1to2_areasum = _cube_area_sum(c1toc2)
                dprint(debug_prefix + 'c1, c1toc2 area-sums=',
                       c1_areasum, c1to2_areasum)
                dprint(debug_prefix + 'REL-DIFF c1/c1toc2 area-sums',
                       _reldiff(c1_areasum, c1to2_areasum))
                self.assertArrayAllClose(c1_areasum, c1to2_areasum, rtol=0.003)


if __name__ == '__main__':
    tests.main()

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
