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
# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import ESMF

import cartopy.crs as ccrs
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.tests.stock as istk

from iris.experimental.regrid import regrid_conservative_with_esmpy

def _generate_test_cubes():
    # create source test cube on rotated form
    pole_lat = 53.4
    pole_lon = -173.2
    deg_swing = 35.3
    pole_lon += deg_swing
#    c1_xvals = np.arange(-30.0, 30.1, 10.0) - deg_swing
#    c1_yvals = np.arange(-20.0, 20.1, 10.0)
    c1_nx = 7
    c1_ny = 5
    c1_xlims = -60.0, 60.0
    c1_ylims = -30.0, 30.0
    scale_x = 1.0
    scale_y = 1.0
    c1_xlims = [x*scale_x for x in c1_xlims]
    c1_ylims = [y*scale_y for y in c1_ylims]
    do_wrapped = False
    do_wrapped = True
    if do_wrapped:
        c1_xlims = [x+360.0 for x in c1_xlims]
    c1_xvals = np.linspace(c1_xlims[0], c1_xlims[1], c1_nx)
    c1_yvals = np.linspace(c1_ylims[0], c1_ylims[1], c1_ny)
    c1_xvals -= deg_swing  # NB *approx* kludge !!  Only if close to -180.
    c1_data = np.array([
        [100, 100, 100, 100, 100, 100, 100],
        [100, 199, 199, 199, 199, 100, 100],
        [100, 100, 100, 199, 199, 100, 100],
        [100, 100, 100, 199, 199, 199, 100],
        [100, 100, 100, 100, 100, 100, 100],
        ],
        dtype=np.float)

    c1 = iris.cube.Cube(c1_data)
    c1_cs = iris.coord_systems.RotatedGeogCS(grid_north_pole_latitude=pole_lat, grid_north_pole_longitude=pole_lon)
    c1_co_x = iris.coords.DimCoord(c1_xvals,
                                   standard_name='grid_longitude',
                                   units=iris.unit.Unit('degrees'),
                                   coord_system=c1_cs)
    c1.add_dim_coord(c1_co_x, 1)
    c1_co_y = iris.coords.DimCoord(c1_yvals,
                                   standard_name='grid_latitude',
                                   units=iris.unit.Unit('degrees'),
                                   coord_system=c1_cs)
    c1.add_dim_coord(c1_co_y, 0)

    # construct target cube to receive
    nx2 = 10
    ny2 = 8
    c2_xlims = -150.0, 200.0
    c2_ylims = -60.0, 90.0
    do_min_covered = False
#    do_min_covered = True
    if do_min_covered:
        # this fixes the no-source-cells error problem
        c2_xlims = -60.0, 90.0
        c2_ylims = -10.0, 80.0
    do_global = False
#    do_global = True
    if do_global:
        nx2 = 60
        ny2 = 40
        dx = 360.0/nx2
        dy = 180.0/ny2
        c2_xlims = -180.0 + 0.5 * dx, 180.0 - 0.5 * dx
        c2_ylims = -90.0 + 0.5 * dy, 90.0 - 0.5 * dy
#    c2_xvals = np.arange(-45.0, 45.1, 10.0) # nx2=10
#    c2_yvals = np.arange(-10.0, 60.1, 10.0) # nx2=8
    c2_xvals = np.linspace(c2_xlims[0], c2_xlims[1], nx2, endpoint=True)
    c2_yvals = np.linspace(c2_ylims[0], c2_ylims[1], ny2, endpoint=False)
    print 'c2_yvals:'
    print c2_yvals
    c2 = iris.cube.Cube(np.zeros((len(c2_yvals), len(c2_xvals))))
    c2_cs = iris.coord_systems.GeogCS(6371229)
    c2_co_x = iris.coords.DimCoord(c2_xvals,
                                   standard_name='longitude',
                                   units=iris.unit.Unit('degrees'),
                                   coord_system=c2_cs)
    c2.add_dim_coord(c2_co_x, 1)
    c2_co_y = iris.coords.DimCoord(c2_yvals,
                                   standard_name='latitude',
                                   units=iris.unit.Unit('degrees'),
                                   coord_system=c2_cs)
    c2.add_dim_coord(c2_co_y, 0)
    
    return c1, c2

class TestConservativeRegrid(tests.IrisTest):
    def test_regrid_conservative_with_esmpy(self):
        # initialise ESMF to log errors
        ESMF.Manager(logkind = ESMF.LogKind.SINGLE, debug = True)

        do_tests = True
    #    do_tests = False
        if do_tests:
            c1, c2 = _generate_test_cubes()
        else:
            c1 = istk.realistic_4d()[0, 0]
            c2 = istk.global_pp()
    
        do_blank_data = False
    #    do_blank_data = True
        if do_blank_data:
            c1.data[:] = 200.0
    
        do_dst_nanfill = True
    #    do_dst_nanfill = False
        if do_dst_nanfill:
            c2.data[:] = np.NaN
    
    #    plt.figure()
    ##    plt.axes(projection=ccrs.PlateCarree())
    #    qplt.contourf(c1)
    #    plt.show()
    
    #    c2.data[4,4] = 1.0
    #        # just so we can show something
    #    plt.figure()
    #    qplt.contourf(c2)
    #    plt.show()
    
    #    plt.figure()
    #    iplt.contourf(c2)
    #    iplt.contourf(c1)
    ##    plt.colorbar()
    #    plt.show()
    
    
    
        t_dbg=0
    
        for cube in c1, c2:
            for axis in ('x', 'y'):
                coord = cube.coord(axis=axis)
                if not coord.has_bounds():
                    coord.guess_bounds()
    
        np.set_printoptions(precision=2, suppress=True)
        print 'c2 X bounds:'
        print c2.coord(axis='x').bounds
        print
        print 'c2 Y bounds:'
        print c2.coord(axis='y').bounds
        print
    
        c1_regrid = regrid_conservative_with_esmpy(c1, c2)
    
    
        levels = [1.0, 50, 99, 120, 140, 160, 180]
    
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        iplt.contourf(c2, levels=levels, extend='both')
        cs = iplt.contourf(c1, levels=levels, extend='both')
        plt.colorbar(cs)
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        plt.show(block=False)
    
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        iplt.contourf(c2, levels=levels, extend='both')
        cs = iplt.contourf(c1_regrid, levels=levels, extend='both')
        plt.colorbar(cs)
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        plt.show(block=False)
    
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        iplt.contourf(c2, levels=levels, extend='both')
        cs = iplt.pcolormesh(c1_regrid) #, levels=levels)
        plt.colorbar(cs)
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        plt.show()
    
        t_dbg = 0


if __name__ == '__main__':
    tests.main()
