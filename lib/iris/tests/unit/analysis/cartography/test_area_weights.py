# (C) British Crown Copyright 2015, Met Office
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

"""Unit tests for the `iris.analysis.cartography.area_weights` function"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.

from __future__ import (absolute_import, division, print_function)
import iris.tests as tests
import iris.tests.stock as stock
import iris.analysis.cartography


class TestInvalidUnits(tests.IrisTest):
    def test_latitude_no_units(self):
        cube = stock.lat_lon_cube()
        cube.coord('longitude').guess_bounds()
        cube.coord('latitude').guess_bounds()
        cube.coord('latitude').units = None
        with self.assertRaisesRegexp(ValueError, 'Units of degrees or '
                                                 'radians required'):
            iris.analysis.cartography.area_weights(cube)

    def test_longitude_no_units(self):
        cube = stock.lat_lon_cube()
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
        cube.coord('longitude').units = None
        with self.assertRaisesRegexp(ValueError, 'Units of degrees or '
                                                 'radians required'):
            iris.analysis.cartography.area_weights(cube)

if __name__ == "__main__":
    tests.main()
