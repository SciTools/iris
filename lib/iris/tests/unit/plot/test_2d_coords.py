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
"""Unit tests for handling and plotting of 2-dimensional coordinates"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
import iris.coords as coords

import numpy.ma as ma

from iris.tests.stock import simple_2d_w_multidim_coords as cube_2dcoords
from iris.tests.stock import simple_3d_w_multidim_coords as cube3d_2dcoords
from iris.tests.stock import lat_lon_cube

from orca_utils.plot_testing import testdata_2d_coords as testdata

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


class Test_2d_coords_plot_defn_bound_mode(tests.IrisTest):
    def setUp(self):
        self.multidim_cube = cube_2dcoords()
        self.overspan_cube = cube3d_2dcoords()

        # latlon_2d is a cube with 2d coords, 4 bounds per point,
        # discontiguities in the bounds but masked data at the discontiguities.
        self.latlon_2d = testdata.full2d_global()
        testdata.make_bounds_discontiguous_at_point(self.latlon_2d, 2, 2)

        # # Take a latlon cube with 1D coords, broadcast the coords into 2D
        # # ones, then add ONE of them back into the cube in place of original:
        # single_dims = lat_lon_cube()
        # lon = single_dims.coord('longitude')
        # lat = single_dims.coord('latitude')
        # big_lon, big_lat = testdata.grid_coords_2d_from_1d(lon, lat)
        # mixed_dims = single_dims.copy()
        # mixed_dims.remove_coord(lon)
        # # TODO Fix this coord addition:
        # # When adding an aux_coord, the function '_check_multidim_metadata'
        # # throws an error as it requires coord.shape to be (1, ) instead of
        # # (3, 4) or whatever.
        # mixed_dims.add_aux_coord(big_lon)
        #
        # # mixed_dims is now a cube with 2 1D dim coords and an additional
        # # 2D aux coord.
        # self.mixed_dims = mixed_dims

        self.mode = coords.BOUND_MODE

    def test_2d_coords_identified(self):
        # Test that 2d coords are identified in the plot definition without
        # having to be specified in the args without any errors.
        cube = self.multidim_cube
        defn = iplt._get_plot_defn(cube, mode=self.mode)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])

    def test_2d_coords_custom_picked(self):
        # Test that 2d coords which are specified in the args will be
        # accepted without any errors.
        cube = self.multidim_cube
        defn = iplt._get_plot_defn_custom_coords_picked(cube, ('foo', 'bar'),
                                                        self.mode)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])

    def test_2d_coords_as_integers(self):
        # Test that if you pass in 2d coords as args in the form of integers,
        # they will still be correctly identified without any errors.
        cube = self.multidim_cube
        defn = iplt._get_plot_defn_custom_coords_picked(cube, (0, 1),
                                                        self.mode)
        self.assertEqual([coord for coord in defn.coords],
                         [1, 0])

    def test_total_span_check(self):
        # Test that an error is raised if a user tries to plot a 2d coord
        # against a different coord, making total number of dimensions 3.
        cube = self.overspan_cube
        with self.assertRaises(ValueError):
            iplt._get_plot_defn_custom_coords_picked(cube, ('wibble', 'foo'),
                                                     self.mode)

    # def test_2dcoord_with_1dcoord(self):
    #     # TODO Generate a cube with one 2d coord and one 1d coord
    #     # TODO Try and plot them against each other
    #     # TODO Find out where I can put a catch for this (if necessary)
    #     cube = self.mixed_dims
    #     with self.assertRaises(ValueError):
    #         iplt._get_plot_defn_custom_coords_picked(cube,
    #                                                  ('latitude', 'longitude'),
    #                                                  self.mode)

    def test_map_common_not_enough_bounds(self):
        # Test that a lat-lon cube with 2d coords and 2 bounds per point
        # throws an error in contiguity checks.
        cube = self.multidim_cube
        cube.coord('foo').rename('longitude')
        cube.coord('bar').rename('latitude')
        with self.assertRaises(ValueError):
            plot_defn = iplt._get_plot_defn(cube, self.mode)
            iplt._map_common('pcolor', None, self.mode, cube, plot_defn)

    def test_map_common_2d(self):
        # Test that a cube with 2d coords can be plotted as a map.
        cube = self.latlon_2d
        # Get necessary variables from _get_plot_defn to check that the test
        # case will be accepted by _map_common.
        plot_defn = iplt._get_plot_defn(cube, self.mode)
        result = iplt._map_common('pcolor', None, self.mode, cube, plot_defn)
        self.assertTrue(result)

    # def test_discontiguous_masked(self):
    #     # Test that a contiguity check will raise a warning (not an error) for
    #     # discontiguous bounds but appropriately masked data.
    #     cube = self.latlon_2d
    #     coord = cube.coord('longitude')
    #     msg = 'The bounds of the longitude coordinate are not contiguous.  ' \
    #           'However, data is masked where the discontiguity occurs so ' \
    #           'plotting anyway.'
    #     with self.assertWarnsRegexp(msg):
    #         iplt._check_contiguity_and_bounds(coord, cube.data)

    def test_discontiguous_unmasked(self):
        # Check that an error occurs when the contiguity check finds
        # discontiguous bounds but unmasked data.
        cube = self.latlon_2d
        cube.data.mask = ma.nomask
        coord = cube.coord('longitude')
        with self.assertRaises(ValueError):
            iplt._check_contiguity_and_bounds(coord, cube.data)

    def test_draw_2d_from_bounds(self):
        # Test this function will not raise an error even with our most
        # awkward but supported cube.
        cube = self.latlon_2d
        result = iplt._draw_2d_from_bounds('pcolormesh', cube)
        self.assertTrue(result)
