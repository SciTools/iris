# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.Geostationary` class."""

import cartopy.crs as ccrs
import pytest

from iris.coord_systems import GeogCS, Geostationary
from iris.tests import _shared_utils


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Set everything to non-default values.
        self.latitude_of_projection_origin = 0  # For now, Cartopy needs =0.
        self.longitude_of_projection_origin = 123.0
        self.perspective_point_height = 9999.0
        self.sweep_angle_axis = "x"
        self.false_easting = 100.0
        self.false_northing = -200.0

        self.semi_major_axis = 4000.0
        self.semi_minor_axis = 3900.0
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )

        # Actual and expected coord system can be re-used for
        # Geostationary.test_crs_creation and test_projection_creation.
        self.expected = ccrs.Geostationary(
            central_longitude=self.longitude_of_projection_origin,
            satellite_height=self.perspective_point_height,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            globe=self.globe,
            sweep_axis=self.sweep_angle_axis,
        )
        self.geo_cs = Geostationary(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.perspective_point_height,
            self.sweep_angle_axis,
            self.false_easting,
            self.false_northing,
            self.ellipsoid,
        )

    def test_crs_creation(self):
        res = self.geo_cs.as_cartopy_crs()
        assert res == self.expected

    def test_projection_creation(self):
        res = self.geo_cs.as_cartopy_projection()
        assert res == self.expected

    def test_non_zero_lat(self):
        with pytest.raises(ValueError, match="Non-zero latitude"):
            Geostationary(
                0.1,
                self.longitude_of_projection_origin,
                self.perspective_point_height,
                self.sweep_angle_axis,
                self.false_easting,
                self.false_northing,
                self.ellipsoid,
            )

    def test_invalid_sweep(self):
        with pytest.raises(ValueError, match="Invalid sweep_angle_axis"):
            Geostationary(
                self.latitude_of_projection_origin,
                self.longitude_of_projection_origin,
                self.perspective_point_height,
                "a",
                self.false_easting,
                self.false_northing,
                self.ellipsoid,
            )

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = Geostationary(0, 0, 1000, "y", false_easting=100, false_northing=-200)
        _shared_utils.assert_equal_and_kind(crs.false_easting, 100.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, -200.0)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        _shared_utils.assert_equal_and_kind(crs.false_easting, 0.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, 0.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = Geostationary(0, 0, 1000, "y")
        self._check_crs_defaults(crs)

    def test_optional_args_none(self):
        # Check expected defaults with optional args=None.
        crs = Geostationary(0, 0, 1000, "y", false_easting=None, false_northing=None)
        self._check_crs_defaults(crs)
