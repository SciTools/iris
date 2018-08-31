# (C) British Crown Copyright 2017 - 2018, Met Office
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
"""Unit tests for the `iris.plot._get_plot_defn` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.coords
from iris.tests.stock import simple_2d, simple_2d_w_multidim_coords

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class Test_get_plot_defn(tests.IrisTest):
    def test_axis_order_xy(self):
        cube_xy = simple_2d()
        defn = iplt._get_plot_defn(cube_xy, iris.coords.POINT_MODE)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])

    def test_axis_order_yx(self):
        cube_yx = simple_2d()
        cube_yx.transpose()
        defn = iplt._get_plot_defn(cube_yx, iris.coords.POINT_MODE)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['foo', 'bar'])

    def test_2d_coords(self):
        cube = simple_2d_w_multidim_coords()
        defn = iplt._get_plot_defn(cube, iris.coords.BOUND_MODE)
        self.assertEqual([coord.name() for coord in defn.coords],
                         ['bar', 'foo'])


if __name__ == "__main__":
    tests.main()
