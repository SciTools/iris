# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.AlbersEqualArea` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs

from iris.coord_systems import AlbersEqualArea, GeogCS


class Test_as_cartopy_crs(tests.IrisTest):
    def setUp(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_central_meridian = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.standard_parallels = (-18.0, -36.0)
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.aea_cs = AlbersEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_central_meridian,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            ellipsoid=self.ellipsoid,
        )

    def test_crs_creation(self):
        res = self.aea_cs.as_cartopy_crs()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.AlbersEqualArea(
            self.longitude_of_central_meridian,
            self.latitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            globe=globe,
        )
        self.assertEqual(res, expected)

    def test_fail_too_few_parallels(self):
        emsg = "parallels"
        with self.assertRaisesRegex(ValueError, emsg):
            AlbersEqualArea(standard_parallels=())

    def test_fail_too_many_parallels(self):
        emsg = "parallels"
        with self.assertRaisesRegex(ValueError, emsg):
            AlbersEqualArea(standard_parallels=(1, 2, 3))


class Test_as_cartopy_projection(tests.IrisTest):
    def setUp(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_central_meridian = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.standard_parallels = (-18.0, -36.0)
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.aea_cs = AlbersEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_central_meridian,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            ellipsoid=self.ellipsoid,
        )

    def test_projection_creation(self):
        res = self.aea_cs.as_cartopy_projection()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.AlbersEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_central_meridian,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            globe=globe,
        )
        self.assertEqual(res, expected)


class Test_init_defaults(tests.IrisTest):
    def test_set_optional_args(self):
        # Check that setting optional arguments works as expected.
        crs = AlbersEqualArea(
            longitude_of_central_meridian=123,
            latitude_of_projection_origin=-17,
            false_easting=100,
            false_northing=-200,
            standard_parallels=(-37, 21.4),
        )

        self.assertEqualAndKind(crs.longitude_of_central_meridian, 123.0)
        self.assertEqualAndKind(crs.latitude_of_projection_origin, -17.0)
        self.assertEqualAndKind(crs.false_easting, 100.0)
        self.assertEqualAndKind(crs.false_northing, -200.0)
        self.assertEqual(len(crs.standard_parallels), 2)
        self.assertEqualAndKind(crs.standard_parallels[0], -37.0)
        self.assertEqualAndKind(crs.standard_parallels[1], 21.4)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        self.assertEqualAndKind(crs.longitude_of_central_meridian, 0.0)
        self.assertEqualAndKind(crs.latitude_of_projection_origin, 0.0)
        self.assertEqualAndKind(crs.false_easting, 0.0)
        self.assertEqualAndKind(crs.false_northing, 0.0)
        self.assertEqual(len(crs.standard_parallels), 2)
        self.assertEqualAndKind(crs.standard_parallels[0], 20.0)
        self.assertEqualAndKind(crs.standard_parallels[1], 50.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = AlbersEqualArea()
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = AlbersEqualArea(
            longitude_of_central_meridian=None,
            latitude_of_projection_origin=None,
            standard_parallels=None,
            false_easting=None,
            false_northing=None,
        )
        self._check_crs_defaults(crs)


if __name__ == "__main__":
    tests.main()
