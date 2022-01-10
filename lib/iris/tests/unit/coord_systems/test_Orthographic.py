# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.Orthographic` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs

from iris.coord_systems import GeogCS, Orthographic


class Test_as_cartopy_crs(tests.IrisTest):
    def setUp(self):
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
        self.assertEqual(res, expected)


class Test_as_cartopy_projection(tests.IrisTest):
    def setUp(self):
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
        self.assertEqual(res, expected)


class Test_init_defaults(tests.IrisTest):
    # NOTE: most of the testing for Orthographic.__init__ is elsewhere.
    # This class *only* tests the defaults for optional constructor args.

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = Orthographic(0, 0, false_easting=100, false_northing=-203.7)
        self.assertEqualAndKind(crs.false_easting, 100.0)
        self.assertEqualAndKind(crs.false_northing, -203.7)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        self.assertEqualAndKind(crs.false_easting, 0.0)
        self.assertEqualAndKind(crs.false_northing, 0.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = Orthographic(0, 0)
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = Orthographic(0, 0, false_easting=None, false_northing=None)
        self._check_crs_defaults(crs)


if __name__ == "__main__":
    tests.main()
