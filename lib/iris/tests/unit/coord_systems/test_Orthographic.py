# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.Orthographic` class."""

import cartopy.crs as ccrs
import pytest

from iris.coord_systems import GeogCS, Orthographic
from iris.tests import _shared_utils


class Test_as_cartopy_crs:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_projection_origin = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.ortho_cs = Orthographic(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            ellipsoid=self.ellipsoid,
        )

    def test_crs_creation(self):
        res = self.ortho_cs.as_cartopy_crs()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.Orthographic(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
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
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.ortho_cs = Orthographic(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            ellipsoid=self.ellipsoid,
        )

    def test_projection_creation(self):
        res = self.ortho_cs.as_cartopy_projection()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.Orthographic(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            globe=globe,
        )
        assert res == expected


class Test_init_defaults:
    # NOTE: most of the testing for Orthographic.__init__ is elsewhere.
    # This class *only* tests the defaults for optional constructor args.

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = Orthographic(0, 0, false_easting=100, false_northing=-203.7)
        _shared_utils.assert_equal_and_kind(crs.false_easting, 100.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, -203.7)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        _shared_utils.assert_equal_and_kind(crs.false_easting, 0.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, 0.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = Orthographic(0, 0)
        self._check_crs_defaults(crs)

    def test_optional_args_none(self):
        # Check expected defaults with optional args=None.
        crs = Orthographic(0, 0, false_easting=None, false_northing=None)
        self._check_crs_defaults(crs)
