# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""Unit tests for the `iris.analysis.cartography.area_weights` function"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
import iris.analysis.cartography
import iris.tests.stock as stock


class TestInvalidUnits(tests.IrisTest):
    def test_latitude_no_units(self):
        cube = stock.lat_lon_cube()
        cube.coord("longitude").guess_bounds()
        cube.coord("latitude").guess_bounds()
        cube.coord("latitude").units = None
        with self.assertRaisesRegex(
            ValueError, "Units of degrees or " "radians required"
        ):
            iris.analysis.cartography.area_weights(cube)

    def test_longitude_no_units(self):
        cube = stock.lat_lon_cube()
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
        cube.coord("longitude").units = None
        with self.assertRaisesRegex(
            ValueError, "Units of degrees or " "radians required"
        ):
            iris.analysis.cartography.area_weights(cube)


if __name__ == "__main__":
    tests.main()
