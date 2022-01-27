# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.Mercator` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs

from iris.coord_systems import GeogCS, Mercator


class Test_Mercator__basics(tests.IrisTest):
    def setUp(self):
        self.tm = Mercator(
            longitude_of_projection_origin=90.0,
            ellipsoid=GeogCS(6377563.396, 6356256.909),
        )

    def test_construction(self):
        self.assertXMLElement(self.tm, ("coord_systems", "Mercator.xml"))

    def test_repr(self):
        expected = (
            "Mercator(longitude_of_projection_origin=90.0, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, "
            "semi_minor_axis=6356256.909), "
            "standard_parallel=0.0, "
            "false_easting=0.0, false_northing=0.0)"
        )
        self.assertEqual(expected, repr(self.tm))


class Test_init_defaults(tests.IrisTest):
    def test_set_optional_args(self):
        # Check that setting the optional (non-ellipse) args works.
        crs = Mercator(
            longitude_of_projection_origin=27,
            standard_parallel=157.4,
            false_easting=13,
            false_northing=12,
        )
        self.assertEqualAndKind(crs.longitude_of_projection_origin, 27.0)
        self.assertEqualAndKind(crs.standard_parallel, 157.4)
        self.assertEqualAndKind(crs.false_easting, 13.0)
        self.assertEqualAndKind(crs.false_northing, 12.0)

    def _check_crs_defaults(self, crs):
        # Check for property defaults when no kwargs options were set.
        # NOTE: except ellipsoid, which is done elsewhere.
        self.assertEqualAndKind(crs.longitude_of_projection_origin, 0.0)
        self.assertEqualAndKind(crs.standard_parallel, 0.0)
        self.assertEqualAndKind(crs.false_easting, 0.0)
        self.assertEqualAndKind(crs.false_northing, 0.0)

    def test_no_optional_args(self):
        # Check expected defaults with no optional args.
        crs = Mercator()
        self._check_crs_defaults(crs)

    def test_optional_args_None(self):
        # Check expected defaults with optional args=None.
        crs = Mercator(
            longitude_of_projection_origin=None,
            standard_parallel=None,
            false_easting=None,
            false_northing=None,
        )
        self._check_crs_defaults(crs)


class Test_Mercator__as_cartopy_crs(tests.IrisTest):
    def test_simple(self):
        # Check that a projection set up with all the defaults is correctly
        # converted to a cartopy CRS.
        merc_cs = Mercator()
        res = merc_cs.as_cartopy_crs()
        # expected = ccrs.Mercator(globe=ccrs.Globe())
        expected = ccrs.Mercator(globe=ccrs.Globe(), latitude_true_scale=0.0)
        self.assertEqual(res, expected)

    def test_extra_kwargs(self):
        # Check that a projection with non-default values is correctly
        # converted to a cartopy CRS.
        longitude_of_projection_origin = 90.0
        true_scale_lat = 14.0
        false_easting = 13
        false_northing = 12
        ellipsoid = GeogCS(
            semi_major_axis=6377563.396, semi_minor_axis=6356256.909
        )

        merc_cs = Mercator(
            longitude_of_projection_origin,
            ellipsoid=ellipsoid,
            standard_parallel=true_scale_lat,
            false_easting=false_easting,
            false_northing=false_northing,
        )

        expected = ccrs.Mercator(
            central_longitude=longitude_of_projection_origin,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
            latitude_true_scale=true_scale_lat,
            false_easting=false_easting,
            false_northing=false_northing,
        )

        res = merc_cs.as_cartopy_crs()
        self.assertEqual(res, expected)


class Test_as_cartopy_projection(tests.IrisTest):
    def test_simple(self):
        # Check that a projection set up with all the defaults is correctly
        # converted to a cartopy projection.
        merc_cs = Mercator()
        res = merc_cs.as_cartopy_projection()
        expected = ccrs.Mercator(globe=ccrs.Globe(), latitude_true_scale=0.0)
        self.assertEqual(res, expected)

    def test_extra_kwargs(self):
        longitude_of_projection_origin = 90.0
        true_scale_lat = 14.0
        false_easting = 13
        false_northing = 12
        ellipsoid = GeogCS(
            semi_major_axis=6377563.396, semi_minor_axis=6356256.909
        )

        merc_cs = Mercator(
            longitude_of_projection_origin,
            ellipsoid=ellipsoid,
            standard_parallel=true_scale_lat,
            false_easting=false_easting,
            false_northing=false_northing,
        )

        expected = ccrs.Mercator(
            central_longitude=longitude_of_projection_origin,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
            latitude_true_scale=true_scale_lat,
            false_easting=false_easting,
            false_northing=false_northing,
        )

        res = merc_cs.as_cartopy_projection()
        self.assertEqual(res, expected)


if __name__ == "__main__":
    tests.main()
