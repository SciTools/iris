# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.LambertAzimuthalEqualArea` class."""

import cartopy.crs as ccrs
import pytest

from iris.coord_systems import GeogCS, LambertAzimuthalEqualArea
from iris.tests import _shared_utils


class Test_as_cartopy_crs:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.latitude_of_projection_origin = 90.0
        self.longitude_of_projection_origin = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.laea_cs = LambertAzimuthalEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            ellipsoid=self.ellipsoid,
        )

    def test_crs_creation(self):
        res = self.laea_cs.as_cartopy_crs()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.LambertAzimuthalEqualArea(
            self.longitude_of_projection_origin,
            self.latitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            globe=globe,
        )
        assert res == expected


class Test_as_cartopy_projection:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_projection_origin = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.laea_cs = LambertAzimuthalEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            ellipsoid=self.ellipsoid,
        )

    def test_projection_creation(self):
        res = self.laea_cs.as_cartopy_projection()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.LambertAzimuthalEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            globe=globe,
        )
        assert res == expected


class Test_init_defaults:
    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = LambertAzimuthalEqualArea(
            longitude_of_projection_origin=123,
            latitude_of_projection_origin=-37,
            false_easting=100,
            false_northing=-200,
        )
        _shared_utils.assert_equal_and_kind(crs.longitude_of_projection_origin, 123.0)
        _shared_utils.assert_equal_and_kind(crs.latitude_of_projection_origin, -37.0)
        _shared_utils.assert_equal_and_kind(crs.false_easting, 100.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, -200.0)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        _shared_utils.assert_equal_and_kind(crs.longitude_of_projection_origin, 0.0)
        _shared_utils.assert_equal_and_kind(crs.latitude_of_projection_origin, 0.0)
        _shared_utils.assert_equal_and_kind(crs.false_easting, 0.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, 0.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = LambertAzimuthalEqualArea()
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = LambertAzimuthalEqualArea(
            longitude_of_projection_origin=None,
            latitude_of_projection_origin=None,
            false_easting=None,
            false_northing=None,
        )
        self._check_crs_defaults(crs)
