# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.LambertConformal` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.coord_systems import LambertConformal


class Test_init_defaults(tests.IrisTest):
    # NOTE: most of the testing for LambertConformal is in the legacy test
    # module 'iris.tests.test_coordsystem'.
    # This class *only* tests the defaults for optional constructor args.

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        # (Except secant_latitudes, which are done separately).
        crs = LambertConformal(
            central_lat=25.3,
            central_lon=-172,
            false_easting=100,
            false_northing=-200,
        )
        self.assertEqualAndKind(crs.central_lat, 25.3)
        self.assertEqualAndKind(crs.central_lon, -172.0)
        self.assertEqualAndKind(crs.false_easting, 100.0)
        self.assertEqualAndKind(crs.false_northing, -200.0)

    def test_set_one_parallel(self):
        # Check that setting the optional (non-ellipse) args works.
        # (Except secant_latitudes, which are done separately).
        crs = LambertConformal(secant_latitudes=-44)
        self.assertEqual(len(crs.secant_latitudes), 1)
        self.assertEqualAndKind(crs.secant_latitudes[0], -44.0)

    def test_set_two_parallels(self):
        # Check that setting the optional (non-ellipse) args works.
        # (Except secant_latitudes, which are done separately).
        crs = LambertConformal(secant_latitudes=[43, -7])
        self.assertEqual(len(crs.secant_latitudes), 2)
        self.assertEqualAndKind(crs.secant_latitudes[0], 43.0)
        self.assertEqualAndKind(crs.secant_latitudes[1], -7.0)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        self.assertEqualAndKind(crs.central_lat, 39.0)
        self.assertEqualAndKind(crs.central_lon, -96.0)
        self.assertEqualAndKind(crs.false_easting, 0.0)
        self.assertEqualAndKind(crs.false_northing, 0.0)
        self.assertEqual(len(crs.secant_latitudes), 2)
        self.assertEqualAndKind(crs.secant_latitudes[0], 33.0)
        self.assertEqualAndKind(crs.secant_latitudes[1], 45.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = LambertConformal()
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = LambertConformal(
            central_lat=None,
            central_lon=None,
            false_easting=None,
            false_northing=None,
            secant_latitudes=None,
        )
        self._check_crs_defaults(crs)


if __name__ == "__main__":
    tests.main()
