# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.TransverseMercator` class."""

from iris.coord_systems import TransverseMercator
from iris.tests import _shared_utils


class Test_init_defaults:
    # NOTE: most of the testing for TransverseMercator is in the legacy test
    # module 'iris.tests.test_coordsystem'.
    # This class *only* tests the defaults for optional constructor args.

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = TransverseMercator(
            0,
            50,
            false_easting=100,
            false_northing=-203.7,
            scale_factor_at_central_meridian=1.057,
        )
        _shared_utils.assert_equal_and_kind(crs.false_easting, 100.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, -203.7)
        _shared_utils.assert_equal_and_kind(crs.scale_factor_at_central_meridian, 1.057)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        _shared_utils.assert_equal_and_kind(crs.false_easting, 0.0)
        _shared_utils.assert_equal_and_kind(crs.false_northing, 0.0)
        _shared_utils.assert_equal_and_kind(crs.scale_factor_at_central_meridian, 1.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = TransverseMercator(0, 50)
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = TransverseMercator(
            0,
            50,
            false_easting=None,
            false_northing=None,
            scale_factor_at_central_meridian=None,
        )
        self._check_crs_defaults(crs)
