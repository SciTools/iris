# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.VerticalPerspective` class."""

import cartopy.crs as ccrs
import pytest

from iris.coord_systems import GeogCS, VerticalPerspective
from iris.tests import _shared_utils


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_projection_origin = 0.0
        self.perspective_point_height = 38204820000.0
        self.false_easting = 0.0
        self.false_northing = 0.0

        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )

        # Actual and expected coord system can be re-used for
        # VerticalPerspective.test_crs_creation and test_projection_creation.
        self.expected = ccrs.NearsidePerspective(
            central_longitude=self.longitude_of_projection_origin,
            central_latitude=self.latitude_of_projection_origin,
            satellite_height=self.perspective_point_height,
            false_easting=self.false_easting,
            false_northing=self.false_northing,
            globe=self.globe,
        )
        self.vp_cs = VerticalPerspective(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.perspective_point_height,
            self.false_easting,
            self.false_northing,
            self.ellipsoid,
        )

    def test_crs_creation(self):
        res = self.vp_cs.as_cartopy_crs()
        assert res == self.expected

    def test_projection_creation(self):
        res = self.vp_cs.as_cartopy_projection()
        assert res == self.expected

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = VerticalPerspective(0, 0, 1000, false_easting=100, false_northing=-203.7)
        _shared_utils.assert_equal_and_kind(crs.false_easting, 100.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, -203.7)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        _shared_utils.assert_equal_and_kind(crs.false_easting, 0.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, 0.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = VerticalPerspective(0, 0, 1000)
        self._check_crs_defaults(crs)

    def test_optional_args_none(self):
        # Check expected defaults with optional args=None.
        crs = VerticalPerspective(0, 0, 1000, false_easting=None, false_northing=None)
        self._check_crs_defaults(crs)
