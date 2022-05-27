# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.PolarStereographic` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs

from iris.coord_systems import GeogCS, PolarStereographic


class Test_PolarStereographic__basics(tests.IrisTest):
    def setUp(self):
        self.ps_blank = PolarStereographic(
            central_lat=90.0,
            central_lon=0,
            ellipsoid=GeogCS(6377563.396, 6356256.909),
        )
        self.ps_standard_parallel = PolarStereographic(
            central_lat=90.0,
            central_lon=0,
            true_scale_lat=30,
            ellipsoid=GeogCS(6377563.396, 6356256.909),
        )
        self.ps_scale_factor = PolarStereographic(
            central_lat=90.0,
            central_lon=0,
            scale_factor_at_projection_origin=1.1,
            ellipsoid=GeogCS(6377563.396, 6356256.909),
        )

    def test_construction(self):
        self.assertXMLElement(
            self.ps_blank, ("coord_systems", "PolarStereographic.xml")
        )

    def test_construction_sp(self):
        self.assertXMLElement(
            self.ps_standard_parallel,
            ("coord_systems", "PolarStereographicStandardParallel.xml"),
        )

    def test_construction_sf(self):
        self.assertXMLElement(
            self.ps_scale_factor,
            ("coord_systems", "PolarStereographicScaleFactor.xml"),
        )

    def test_repr_blank(self):
        expected = (
            "PolarStereographic(central_lat=90.0, central_lon=0.0, "
            "false_easting=0.0, false_northing=0.0, "
            "true_scale_lat=None, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, "
            "semi_minor_axis=6356256.909))"
        )
        self.assertEqual(expected, repr(self.ps_blank))

    def test_repr_standard_parallel(self):
        expected = (
            "PolarStereographic(central_lat=90.0, central_lon=0.0, "
            "false_easting=0.0, false_northing=0.0, "
            "true_scale_lat=30.0, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, "
            "semi_minor_axis=6356256.909))"
        )
        self.assertEqual(expected, repr(self.ps_standard_parallel))

    def test_repr_scale_factor(self):
        expected = (
            "PolarStereographic(central_lat=90.0, central_lon=0.0, "
            "false_easting=0.0, false_northing=0.0, "
            "scale_factor_at_projection_origin=1.1, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, "
            "semi_minor_axis=6356256.909))"
        )
        self.assertEqual(expected, repr(self.ps_scale_factor))


class Test_init_defaults(tests.IrisTest):
    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = PolarStereographic(
            central_lat=90,
            central_lon=50,
            false_easting=13,
            false_northing=12,
            true_scale_lat=32,
        )
        self.assertEqualAndKind(crs.central_lat, 90.0)
        self.assertEqualAndKind(crs.central_lon, 50.0)
        self.assertEqualAndKind(crs.false_easting, 13.0)
        self.assertEqualAndKind(crs.false_northing, 12.0)
        self.assertEqualAndKind(crs.true_scale_lat, 32.0)

    def test_set_optional_scale_factor_alternative(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = PolarStereographic(
            central_lat=-90,
            central_lon=50,
            false_easting=13,
            false_northing=12,
            scale_factor_at_projection_origin=3.1,
        )
        self.assertEqualAndKind(crs.central_lat, -90.0)
        self.assertEqualAndKind(crs.central_lon, 50.0)
        self.assertEqualAndKind(crs.false_easting, 13.0)
        self.assertEqualAndKind(crs.false_northing, 12.0)
        self.assertEqualAndKind(crs.scale_factor_at_projection_origin, 3.1)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        self.assertEqualAndKind(crs.false_easting, 0.0)
        self.assertEqualAndKind(crs.false_northing, 0.0)
        self.assertEqualAndKind(crs.true_scale_lat, None)
        self.assertEqualAndKind(crs.scale_factor_at_projection_origin, None)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = PolarStereographic(
            central_lat=-90,
            central_lon=50,
        )
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = PolarStereographic(
            central_lat=-90,
            central_lon=50,
            true_scale_lat=None,
            scale_factor_at_projection_origin=None,
            false_easting=None,
            false_northing=None,
        )
        self._check_crs_defaults(crs)


class AsCartopyMixin:
    def test_simple(self):
        # Check that a projection set up with all the defaults is correctly
        # converted to a cartopy CRS.
        central_lat = -90
        central_lon = 50
        polar_cs = PolarStereographic(
            central_lat=central_lat,
            central_lon=central_lon,
        )
        res = self.as_cartopy_method(polar_cs)
        expected = ccrs.Stereographic(
            central_latitude=central_lat,
            central_longitude=central_lon,
            globe=ccrs.Globe(),
        )
        self.assertEqual(res, expected)

    def test_extra_kwargs_scale_factor(self):
        # Check that a projection with non-default values is correctly
        # converted to a cartopy CRS.
        central_lat = -90
        central_lon = 50
        scale_factor_at_projection_origin = 1.3
        false_easting = 13
        false_northing = 15
        ellipsoid = GeogCS(
            semi_major_axis=6377563.396, semi_minor_axis=6356256.909
        )

        polar_cs = PolarStereographic(
            central_lat=central_lat,
            central_lon=central_lon,
            scale_factor_at_projection_origin=scale_factor_at_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            ellipsoid=ellipsoid,
        )

        expected = ccrs.Stereographic(
            central_latitude=central_lat,
            central_longitude=central_lon,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor=scale_factor_at_projection_origin,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
        )

        res = self.as_cartopy_method(polar_cs)
        self.assertEqual(res, expected)

    def test_extra_kwargs_true_scale_lat_alternative(self):
        # Check that a projection with non-default values is correctly
        # converted to a cartopy CRS.
        central_lat = -90
        central_lon = 50
        true_scale_lat = 80
        false_easting = 13
        false_northing = 15
        ellipsoid = GeogCS(
            semi_major_axis=6377563.396, semi_minor_axis=6356256.909
        )

        polar_cs = PolarStereographic(
            central_lat=central_lat,
            central_lon=central_lon,
            true_scale_lat=true_scale_lat,
            false_easting=false_easting,
            false_northing=false_northing,
            ellipsoid=ellipsoid,
        )

        expected = ccrs.Stereographic(
            central_latitude=central_lat,
            central_longitude=central_lon,
            false_easting=false_easting,
            false_northing=false_northing,
            true_scale_latitude=true_scale_lat,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
        )

        res = self.as_cartopy_method(polar_cs)
        self.assertEqual(res, expected)


class Test_PolarStereographic__as_cartopy_crs(tests.IrisTest, AsCartopyMixin):
    def setUp(self):
        self.as_cartopy_method = PolarStereographic.as_cartopy_crs


class Test_PolarStereographic__as_cartopy_projection(
    tests.IrisTest, AsCartopyMixin
):
    def setUp(self):
        self.as_cartopy_method = PolarStereographic.as_cartopy_projection


if __name__ == "__main__":
    tests.main()
