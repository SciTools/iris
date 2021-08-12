# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""Unit tests for the `iris.analysis.cartography._quadrant_area` function"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests  # isort:skip

import cf_units
import numpy as np

from iris.analysis.cartography import (
    DEFAULT_SPHERICAL_EARTH_RADIUS,
    _quadrant_area,
)


class TestExampleCases(tests.IrisTest):
    def _radian_bounds(self, coord_list, dtype):
        bound_deg = np.array(coord_list, dtype=dtype)
        bound_deg = np.atleast_2d(bound_deg)
        degrees = cf_units.Unit("degrees")
        radians = cf_units.Unit("radians")
        return degrees.convert(bound_deg, radians)

    def _as_bounded_coords(self, lats, lons, dtype=np.float64):
        return (
            self._radian_bounds(lats, dtype=dtype),
            self._radian_bounds(lons, dtype=dtype),
        )

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
        lats, lons = self._as_bounded_coords(
            [[-80, -70], [0, 10], [70, 80]], [0, 10]
        )
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)

        self.assertArrayAllClose(
            area,
            [
                [319251845980.763671875],
                [1228800593851.443115234375],
                [319251845980.7646484375],
            ],
        )

    def test_area_multiple_lats_and_lons(self):
        lats, lons = self._as_bounded_coords(
            [[-80, -70], [0, 10], [70, 80]], [[0, 10], [10, 30]]
        )
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)

        self.assertArrayAllClose(
            area,
            [
                [3.19251846e11, 6.38503692e11],
                [1.22880059e12, 2.45760119e12],
                [3.19251846e11, 6.38503692e11],
            ],
        )

    def test_symmetric_64_bit(self):
        lats, lons = self._as_bounded_coords(
            [[-90, -89.375], [89.375, 90]], [0, 10], dtype=np.float64
        )
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertArrayAllClose(area, area[::-1])

    def test_symmetric_32_bit(self):
        lats, lons = self._as_bounded_coords(
            [[-90, -89.375], [89.375, 90]], [0, 10], dtype=np.float32
        )
        area = _quadrant_area(lats, lons, DEFAULT_SPHERICAL_EARTH_RADIUS)
        self.assertArrayAllClose(area, area[::-1])


class TestErrorHandling(tests.IrisTest):
    def test_lat_bounds_1d_error(self):
        self._assert_error_on_malformed_bounds([0, 10], [[0, 10]])

    def test_lon_bounds_1d_error(self):
        self._assert_error_on_malformed_bounds([[0, 10]], [0, 10])

    def test_too_many_lat_bounds_error(self):
        self._assert_error_on_malformed_bounds([[0, 10, 20]], [[0, 10]])

    def test_too_many_lon_bounds_error(self):
        self._assert_error_on_malformed_bounds([[0, 10]], [[0, 10, 20]])

    def _assert_error_on_malformed_bounds(self, lat_bnds, lon_bnds):
        with self.assertRaisesRegex(
            ValueError, r"Bounds must be \[n,2\] array"
        ):
            _quadrant_area(np.array(lat_bnds), np.array(lon_bnds), 1.0)


if __name__ == "__main__":
    tests.main()
