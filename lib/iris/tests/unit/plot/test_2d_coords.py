# (C) British Crown Copyright 2014 - 2016, Met Office
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

import numpy as np

from iris.tests.stock import simple_2d_w_multidim_coords as cube_2dcoords
from iris.tests.stock import simple_3d_w_multidim_coords as cube3d_2dcoords
from iris.tests.stock import discontiguous_bounds_2dcoords

from orca_utils import load_standard_testdata

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


# TODO Integration graphics test

# TODO Test _draw_2d_from_bounds
# TODO Test _draw_2d_from_points (later)

class Test_2d_coords_plot_defn_bound_mode(tests.IrisTest):
    def setUp(self):
        self.multidim_cube = cube_2dcoords()
        self.overspan_cube = cube3d_2dcoords()
        self.discontiguous_cube = discontiguous_bounds_2dcoords()
        # TODO add latlon cube (with coords lat and lon) to test_map_common
        self.mode = coords.BOUND_MODE

    def test_2d_coords_identified(self):
        defn = iplt._get_plot_defn(self.multidim_cube, mode=self.mode)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])

    def test_2d_coords_custom_picked(self):
        defn = iplt._get_plot_defn_custom_coords_picked(self.multidim_cube,
                                                        ('foo', 'bar'),
                                                        self.mode)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])

    def test_2d_coords_as_integers(self):
        defn = iplt._get_plot_defn_custom_coords_picked(self.multidim_cube,
                                                        (0, 1),
                                                        self.mode)
        self.assertEqual([coord for coord in defn.coords],
                         [1, 0])

    def test_total_span_check(self):
        with self.assertRaises(ValueError):
            iplt._get_plot_defn_custom_coords_picked(self.overspan_cube,
                                                     ('wibble', 'foo'),
                                                     self.mode)

    def test_map_common(self):
        # Test with a lat-lon cube with 2d coords and 4 bounds per point.
        cube = self.multidim_cube
        cube.coord('foo').rename('longitude')
        cube.coord('bar').rename('latitude')

        plot_defn = iplt._get_plot_defn(cube, self.mode)
        result = iplt._map_common('pcolor', None, self.mode, cube, plot_defn)
        # Complexity of 'result' made me think maybe we should just test that
        # it doesn't fail with 2d coords
        self.assertTrue(result)

    def test_discontiguous_masked(self):
        cube = self.discontiguous_cube
        coord = cube.coord('foo')
        expected_msg = 'The bounds of the foo coordinate are not ' \
                       'contiguous.  However, data is masked where the ' \
                       'discontiguity occurs so plotting anyway.'
        with self.assertWarnsRegexp(expected_msg):
            iplt._check_contiguity_and_bounds(coord, cube.data)

    def test_discontiguous_unmasked(self):
        cube = self.discontiguous_cube
        # This should raise an error
            
    def test_draw_2d_from_bounds(self):
        pass


# class Test_2d_coords_plot_defn_point_mode(tests.IrisTest):
#     def setUp(self):
#         self.multidim_cube = cube_2dcoords()
#         self.overspan_cube = cube3d_2dcoords()
#         self.mode = coords.POINTS_MODE
# 
#     # TODO Test custom coords for POINTS_MODE?
#     # TODO Test total span check for POINTS_MODE


