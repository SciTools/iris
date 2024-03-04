# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.Stereographic` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs

from iris.coord_systems import GeogCS, Stereographic


def stereo(**kwargs):
    return Stereographic(
        central_lat=-90,
        central_lon=-45,
        false_easting=100,
        false_northing=200,
        ellipsoid=GeogCS(6377563.396, 6356256.909),
        **kwargs,
    )


class Test_Stereographic_construction(tests.IrisTest):
    def test_stereo(self):
        st = stereo()
        self.assertXMLElement(st, ("coord_systems", "Stereographic.xml"))


class Test_init_defaults(tests.IrisTest):
    # This class *only* tests the defaults for optional constructor args.

    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = Stereographic(
            0, 0, false_easting=100, false_northing=-203.7, true_scale_lat=77
        )
        self.assertEqualAndKind(crs.false_easting, 100.0)
        self.assertEqualAndKind(crs.false_northing, -203.7)
        self.assertEqualAndKind(crs.true_scale_lat, 77.0)

    def test_set_optional_args_scale_factor_alternative(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = Stereographic(
            0,
            0,
            false_easting=100,
            false_northing=-203.7,
            scale_factor_at_projection_origin=1.3,
        )
        self.assertEqualAndKind(crs.false_easting, 100.0)
        self.assertEqualAndKind(crs.false_northing, -203.7)
        self.assertEqualAndKind(crs.scale_factor_at_projection_origin, 1.3)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        self.assertEqualAndKind(crs.false_easting, 0.0)
        self.assertEqualAndKind(crs.false_northing, 0.0)
        self.assertIsNone(crs.true_scale_lat)
        self.assertIsNone(crs.scale_factor_at_projection_origin)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = Stereographic(0, 0)
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = Stereographic(
            0,
            0,
            false_easting=None,
            false_northing=None,
            true_scale_lat=None,
            scale_factor_at_projection_origin=None,
        )
        self._check_crs_defaults(crs)


class Test_Stereographic_repr(tests.IrisTest):
    def test_stereo(self):
        st = stereo()
        expected = (
            "Stereographic(central_lat=-90.0, central_lon=-45.0, "
            "false_easting=100.0, false_northing=200.0, true_scale_lat=None, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909))"
        )
        self.assertEqual(expected, repr(st))

    def test_stereo_scale_factor(self):
        st = stereo(scale_factor_at_projection_origin=0.9)
        expected = (
            "Stereographic(central_lat=-90.0, central_lon=-45.0, "
            "false_easting=100.0, false_northing=200.0, "
            "scale_factor_at_projection_origin=0.9, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909))"
        )
        self.assertEqual(expected, repr(st))


class AsCartopyMixin:
    def test_basic(self):
        latitude_of_projection_origin = -90.0
        longitude_of_projection_origin = -45.0
        false_easting = 100.0
        false_northing = 200.0
        ellipsoid = GeogCS(6377563.396, 6356256.909)

        st = Stereographic(
            central_lat=latitude_of_projection_origin,
            central_lon=longitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            ellipsoid=ellipsoid,
        )
        expected = ccrs.Stereographic(
            central_latitude=latitude_of_projection_origin,
            central_longitude=longitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
        )

        res = self.as_cartopy_method(st)
        self.assertEqual(res, expected)

    def test_true_scale_lat(self):
        latitude_of_projection_origin = -90.0
        longitude_of_projection_origin = -45.0
        false_easting = 100.0
        false_northing = 200.0
        true_scale_lat = 30
        ellipsoid = GeogCS(6377563.396, 6356256.909)

        st = Stereographic(
            central_lat=latitude_of_projection_origin,
            central_lon=longitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            true_scale_lat=true_scale_lat,
            ellipsoid=ellipsoid,
        )
        expected = ccrs.Stereographic(
            central_latitude=latitude_of_projection_origin,
            central_longitude=longitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            true_scale_latitude=true_scale_lat,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
        )

        res = self.as_cartopy_method(st)
        self.assertEqual(res, expected)

    def test_scale_factor(self):
        latitude_of_projection_origin = -90.0
        longitude_of_projection_origin = -45.0
        false_easting = 100.0
        false_northing = 200.0
        scale_factor_at_projection_origin = 0.9
        ellipsoid = GeogCS(6377563.396, 6356256.909)

        st = Stereographic(
            central_lat=latitude_of_projection_origin,
            central_lon=longitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor_at_projection_origin=scale_factor_at_projection_origin,
            ellipsoid=ellipsoid,
        )
        expected = ccrs.Stereographic(
            central_latitude=latitude_of_projection_origin,
            central_longitude=longitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor=scale_factor_at_projection_origin,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
        )

        res = self.as_cartopy_method(st)
        self.assertEqual(res, expected)


class Test_Stereographic_as_cartopy_crs(tests.IrisTest, AsCartopyMixin):
    def setUp(self):
        self.as_cartopy_method = Stereographic.as_cartopy_crs


class Test_Stereographic_as_cartopy_projection(tests.IrisTest, AsCartopyMixin):
    def setUp(self):
        self.as_cartopy_method = Stereographic.as_cartopy_projection


if __name__ == "__main__":
    tests.main()
