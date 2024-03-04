# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot._get_plot_defn` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import iris.coords
from iris.tests.stock import simple_2d, simple_2d_w_multidim_coords

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class Test_get_plot_defn(tests.IrisTest):
    def test_axis_order_xy(self):
        cube_xy = simple_2d()
        defn = iplt._get_plot_defn(cube_xy, iris.coords.POINT_MODE)
        self.assertEqual(
            [coord.name() for coord in defn.coords], ["bar", "foo"]
        )

    def test_axis_order_yx(self):
        cube_yx = simple_2d()
        cube_yx.transpose()
        defn = iplt._get_plot_defn(cube_yx, iris.coords.POINT_MODE)
        self.assertEqual(
            [coord.name() for coord in defn.coords], ["foo", "bar"]
        )

    def test_2d_coords(self):
        cube = simple_2d_w_multidim_coords()
        defn = iplt._get_plot_defn(cube, iris.coords.BOUND_MODE)
        self.assertEqual(
            [coord.name() for coord in defn.coords], ["bar", "foo"]
        )


if __name__ == "__main__":
    tests.main()
