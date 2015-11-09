# (C) British Crown Copyright 2014 - 2015, Met Office
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

"""Unit tests for the `iris.analysis.cartography._quadrant_area` function"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests

import cf_units
import numpy as np

import iris
from iris.analysis.cartography import _quadrant_area
from iris.analysis.cartography import DEFAULT_SPHERICAL_EARTH_RADIUS


class TestExampleCases(tests.IrisTest):

    def _radian_bounds(self, coord_list, offset=0):
        bound_deg = np.array(coord_list) + offset
        bound_deg = np.atleast_2d(bound_deg)
        degrees = cf_units.Unit("degrees")
        radians = cf_units.Unit("radians")
        return degrees.convert(bound_deg, radians)

    def _as_bounded_coords(self, lats, lons):
        return (self._radian_bounds(lats, offset=90),
                self._radian_bounds(lons))

    def test_area_in_north(self):
        lats, lons = self._as_bounded_coords([0, 10], [0, 10])
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertArrayAllClose(area, [[1228800593851.443115234375]])

    def test_area_in_far_north(self):
        lats, lons = self._as_bounded_coords([70, 80], [0, 10])
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertArrayAllClose(area, [[319251845980.7646484375]])

    def test_area_in_far_south(self):
        lats, lons = self._as_bounded_coords([-80, -70], [0, 10])
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertArrayAllClose(area, [[319251845980.763671875]])

    def test_area_in_north_with_reversed_lats(self):
        lats, lons = self._as_bounded_coords([10, 0], [0, 10])
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertArrayAllClose(area, [[1228800593851.443115234375]])

    def test_area_multiple_lats(self):
        lats, lons = self._as_bounded_coords([[-80, -70], [0, 10], [70, 80]],
                                             [0, 10])
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)

        self.assertArrayAllClose(area, [[319251845980.763671875],
                                        [1228800593851.443115234375],
                                        [319251845980.7646484375]])

    def test_area_multiple_lats_and_lons(self):
        lats, lons = self._as_bounded_coords([[-80, -70], [0, 10], [70, 80]],
                                             [[0, 10], [10, 30]])
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)

        self.assertArrayAllClose(area, [[3.19251846e+11, 6.38503692e+11],
                                        [1.22880059e+12, 2.45760119e+12],
                                        [3.19251846e+11, 6.38503692e+11]])


class TestErrorHandling(tests.IrisTest):

    def test_lat_bounds_1d_error(self):
        self._assert_error_on_malformed_bounds(
            [0, 10],
            [[0, 10]])

    def test_lon_bounds_1d_error(self):
        self._assert_error_on_malformed_bounds(
            [[0, 10]],
            [0, 10])

    def test_too_many_lat_bounds_error(self):
        self._assert_error_on_malformed_bounds(
            [[0, 10, 20]],
            [[0, 10]])

    def test_too_many_lon_bounds_error(self):
        self._assert_error_on_malformed_bounds(
            [[0, 10]],
            [[0, 10, 20]])

    def _assert_error_on_malformed_bounds(self, lat_bnds, lon_bnds):
        with self.assertRaisesRegexp(ValueError,
                                     'Bounds must be \[n,2\] array'):
            _quadrant_area(np.array(lat_bnds),
                           np.array(lon_bnds),
                           1.)

if __name__ == '__main__':
    tests.main()
