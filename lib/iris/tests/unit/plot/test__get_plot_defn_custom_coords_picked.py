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
"""Unit tests for the `iris.plot._get_plot_defn_custom_coords_picked`
function."""


from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.coords import BOUND_MODE, POINT_MODE
from iris.tests.stock import (simple_2d, simple_2d_w_multidim_coords,
                              hybrid_height)


if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class Test_get_plot_defn_custom_coords_picked(tests.IrisTest):
    def test_1d_coords(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, ('foo', 'bar'),
                                                        POINT_MODE)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])

    def test_1d_coords_swapped(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, ('bar', 'foo'),
                                                        POINT_MODE)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['foo', 'bar'])

    def test_1d_coords_as_integers(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, (0, 1),
                                                        POINT_MODE)
        self.assertEqual([coord for coord in defn.coords], [1, 0])

    def test_1d_coords_as_integers_swapped(self):
        cube = simple_2d()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, (1, 0),
                                                        POINT_MODE)
        self.assertEqual([coord for coord in defn.coords], [0, 1])

    def test_2d_coords(self):
        cube = simple_2d_w_multidim_coords()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, ('foo', 'bar'),
                                                        BOUND_MODE)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])

    def test_2d_coords_as_integers(self):
        cube = simple_2d_w_multidim_coords()
        defn = iplt._get_plot_defn_custom_coords_picked(cube, (0, 1),
                                                        BOUND_MODE)
        self.assertEqual([coord for coord in defn.coords], [1, 0])

    def test_span_check(self):
        cube = hybrid_height()
        emsg = 'The given coordinates (sigma, level_height) don\'t span the ' \
               '2 data dimensions.'

        with self.assertRaisesRegexp(ValueError, emsg):
            iplt._get_plot_defn_custom_coords_picked(
                cube, ('sigma', 'level_height'), POINT_MODE)

    def test_2dcoord_with_1dcoord(self):
        cube = hybrid_height()
        emsg = 'Cell-based plotting is only supported for ' \
               'coordinates with the same number of dimensions'
        with self.assertRaisesRegexp(ValueError, emsg):
            iplt._get_plot_defn_custom_coords_picked(
                cube, ('level_height', 'altitude'), BOUND_MODE)


if __name__ == "__main__":
    tests.main()

