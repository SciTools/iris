# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.RotatedPole` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import cartopy
import cartopy.crs as ccrs

from iris.coord_systems import RotatedGeogCS


class Test_init(tests.IrisTest):
    def setUp(self):
        self.pole_lon = 171.77
        self.pole_lat = 49.55
        self.rotation_about_new_pole = 180.0
        self.rp_crs = RotatedGeogCS(
            self.pole_lat, self.pole_lon, self.rotation_about_new_pole
        )

    def test_crs_creation(self):
        self.assertEqual(self.pole_lon, self.rp_crs.grid_north_pole_longitude)
        self.assertEqual(self.pole_lat, self.rp_crs.grid_north_pole_latitude)
        self.assertEqual(
            self.rotation_about_new_pole, self.rp_crs.north_pole_grid_longitude
        )

    def test_as_cartopy_crs(self):
        if cartopy.__version__ < "0.12":
            with mock.patch("warnings.warn") as warn:
                accrs = self.rp_crs.as_cartopy_crs()
                self.assertEqual(warn.call_count, 1)
        else:
            accrs = self.rp_crs.as_cartopy_crs()
            expected = ccrs.RotatedGeodetic(
                self.pole_lon, self.pole_lat, self.rotation_about_new_pole
            )
            self.assertEqual(
                sorted(accrs.proj4_init.split(" +")),
                sorted(expected.proj4_init.split(" +")),
            )

    def test_as_cartopy_projection(self):
        if cartopy.__version__ < "0.12":
            with mock.patch("warnings.warn") as warn:
                _ = self.rp_crs.as_cartopy_projection()
                self.assertEqual(warn.call_count, 1)
        else:
            accrsp = self.rp_crs.as_cartopy_projection()
            expected = ccrs.RotatedPole(
                self.pole_lon, self.pole_lat, self.rotation_about_new_pole
            )
            self.assertEqual(
                sorted(accrsp.proj4_init.split(" +")),
                sorted(expected.proj4_init.split(" +")),
            )

    def _check_crs_default(self, crs):
        # Check for property defaults when no kwargs options are set.
        # NOTE: except ellipsoid, which is done elsewhere.
        self.assertEqualAndKind(crs.north_pole_grid_longitude, 0.0)

    def test_optional_args_missing(self):
        # Check that unused 'north_pole_grid_longitude' defaults to 0.0.
        crs = RotatedGeogCS(self.pole_lon, self.pole_lat)
        self._check_crs_default(crs)

    def test_optional_args_None(self):
        # Check that 'north_pole_grid_longitude=None' defaults to 0.0.
        crs = RotatedGeogCS(
            self.pole_lon, self.pole_lat, north_pole_grid_longitude=None
        )
        self._check_crs_default(crs)


if __name__ == "__main__":
    tests.main()
