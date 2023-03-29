# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.GeogCS` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.coord_systems import GeogCS


class Test_init_defaults(tests.IrisTest):
    # NOTE: most of the testing for GeogCS is in the legacy test module
    # 'iris.tests.test_coordsystem'.
    # This class *only* tests the defaults for optional constructor args.

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) argument works.
        crs = GeogCS(1.0, longitude_of_prime_meridian=-85)
        self.assertEqualAndKind(crs.longitude_of_prime_meridian, -85.0)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        radius = float(crs.semi_major_axis)
        self.assertEqualAndKind(crs.semi_major_axis, radius)  # just the kind
        self.assertEqualAndKind(crs.semi_minor_axis, radius)
        self.assertEqualAndKind(crs.inverse_flattening, 0.0)
        self.assertEqualAndKind(crs.longitude_of_prime_meridian, 0.0)

    def test_no_optional_args(self):
        # Check expected properties with no optional args.
        crs = GeogCS(1.0)
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected properties with optional args=None.
        crs = GeogCS(
            1.0,
            semi_minor_axis=None,
            inverse_flattening=None,
            longitude_of_prime_meridian=None,
        )
        self._check_crs_defaults(crs)

    def test_zero_inverse_flattening_on_perfect_sphere(self):
        # allow inverse_flattening to be 0 for a perfect sphere
        # i.e. semi-major axis defined, semi-minor is None.
        crs = GeogCS(
            1.0,
            semi_minor_axis=None,
            inverse_flattening=0.0,
            longitude_of_prime_meridian=None,
        )
        self._check_crs_defaults(crs)


if __name__ == "__main__":
    tests.main()
