# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip


import cartopy.crs as ccrs

from iris.coord_systems import (
    GeogCS,
    LambertConformal,
    RotatedGeogCS,
    Stereographic,
    TransverseMercator,
)
import iris.coords
import iris.cube
import iris.tests.stock


def osgb():
    return TransverseMercator(
        latitude_of_projection_origin=49,
        longitude_of_central_meridian=-2,
        false_easting=-400,
        false_northing=100,
        scale_factor_at_central_meridian=0.9996012717,
        ellipsoid=GeogCS(6377563.396, 6356256.909),
    )


def stereo():
    return Stereographic(
        central_lat=-90,
        central_lon=-45,
        false_easting=100,
        false_northing=200,
        ellipsoid=GeogCS(6377563.396, 6356256.909),
    )


class TestCoordSystemLookup(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.lat_lon_cube()

    def test_hit_name(self):
        self.assertIsInstance(self.cube.coord_system("GeogCS"), GeogCS)

    def test_hit_type(self):
        self.assertIsInstance(self.cube.coord_system(GeogCS), GeogCS)

    def test_miss(self):
        self.assertIsNone(self.cube.coord_system(RotatedGeogCS))

    def test_empty(self):
        self.assertIsInstance(self.cube.coord_system(GeogCS), GeogCS)
        self.assertIsNotNone(self.cube.coord_system(None))
        self.assertIsInstance(self.cube.coord_system(None), GeogCS)
        self.assertIsNotNone(self.cube.coord_system())
        self.assertIsInstance(self.cube.coord_system(), GeogCS)

        for coord in self.cube.coords():
            coord.coord_system = None

        self.assertIsNone(self.cube.coord_system(GeogCS))
        self.assertIsNone(self.cube.coord_system(None))
        self.assertIsNone(self.cube.coord_system())


class TestCoordSystemSame(tests.IrisTest):
    def setUp(self):
        self.cs1 = iris.coord_systems.GeogCS(6371229)
        self.cs2 = iris.coord_systems.GeogCS(6371229)
        self.cs3 = iris.coord_systems.RotatedGeogCS(
            30, 30, ellipsoid=GeogCS(6371229)
        )

    def test_simple(self):
        a = self.cs1
        b = self.cs2
        self.assertEqual(a, b)

    def test_different_class(self):
        a = self.cs1
        b = self.cs3
        self.assertNotEquals(a, b)

    def test_different_public_attributes(self):
        a = self.cs1
        b = self.cs2
        a.foo = "a"

        # check that that attribute was added (just in case)
        self.assertEqual(a.foo, "a")

        # a and b should not be the same
        self.assertNotEquals(a, b)

        # a and b should be the same
        b.foo = "a"
        self.assertEqual(a, b)

        b.foo = "b"
        # a and b should not be the same
        self.assertNotEquals(a, b)


class Test_CoordSystem_xml_element(tests.IrisTest):
    def test_rotated(self):
        cs = RotatedGeogCS(30, 40, ellipsoid=GeogCS(6371229))
        self.assertXMLElement(
            cs, ("coord_systems", "CoordSystem_xml_element.xml")
        )


class Test_GeogCS_construction(tests.IrisTest):
    # Test Ellipsoid constructor
    # Don't care about testing the units, it has no logic specific to this class.

    def test_sphere_param(self):
        cs = GeogCS(6543210)
        self.assertXMLElement(cs, ("coord_systems", "GeogCS_init_sphere.xml"))

    def test_no_major(self):
        cs = GeogCS(
            semi_minor_axis=6500000, inverse_flattening=151.42814163388104
        )
        self.assertXMLElement(
            cs, ("coord_systems", "GeogCS_init_no_major.xml")
        )

    def test_no_minor(self):
        cs = GeogCS(
            semi_major_axis=6543210, inverse_flattening=151.42814163388104
        )
        self.assertXMLElement(
            cs, ("coord_systems", "GeogCS_init_no_minor.xml")
        )

    def test_no_invf(self):
        cs = GeogCS(semi_major_axis=6543210, semi_minor_axis=6500000)
        self.assertXMLElement(cs, ("coord_systems", "GeogCS_init_no_invf.xml"))

    def test_invalid_ellipsoid_params(self):
        # no params
        with self.assertRaises(ValueError):
            GeogCS()

        # over specified
        with self.assertRaises(ValueError):
            GeogCS(6543210, 6500000, 151.42814163388104)

        # under specified
        with self.assertRaises(ValueError):
            GeogCS(None, 6500000, None)
        with self.assertRaises(ValueError):
            GeogCS(None, None, 151.42814163388104)


class Test_GeogCS_repr(tests.IrisTest):
    def test_repr(self):
        cs = GeogCS(6543210, 6500000)
        expected = (
            "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0)"
        )
        self.assertEqual(expected, repr(cs))


class Test_GeogCS_str(tests.IrisTest):
    def test_str(self):
        cs = GeogCS(6543210, 6500000)
        expected = (
            "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0)"
        )
        self.assertEqual(expected, str(cs))


class Test_GeogCS_as_cartopy_globe(tests.IrisTest):
    def test_as_cartopy_globe(self):
        cs = GeogCS(6543210, 6500000)
        # Can't check equality directly, so use the proj4 params instead.
        res = cs.as_cartopy_globe().to_proj4_params()
        expected = {"a": 6543210, "b": 6500000}
        self.assertEqual(res, expected)


class Test_GeogCS_as_cartopy_crs(tests.IrisTest):
    def test_as_cartopy_crs(self):
        cs = GeogCS(6543210, 6500000)
        res = cs.as_cartopy_crs()
        globe = ccrs.Globe(
            semimajor_axis=6543210.0, semiminor_axis=6500000.0, ellipse=None
        )
        expected = ccrs.Geodetic(globe)
        self.assertEqual(res, expected)


class Test_RotatedGeogCS_construction(tests.IrisTest):
    def test_init(self):
        rcs = RotatedGeogCS(
            30, 40, north_pole_grid_longitude=50, ellipsoid=GeogCS(6371229)
        )
        self.assertXMLElement(rcs, ("coord_systems", "RotatedGeogCS_init.xml"))

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        self.assertXMLElement(
            rcs, ("coord_systems", "RotatedGeogCS_init_a.xml")
        )

        rcs = RotatedGeogCS(30, 40)
        self.assertXMLElement(
            rcs, ("coord_systems", "RotatedGeogCS_init_b.xml")
        )


class Test_RotatedGeogCS_repr(tests.IrisTest):
    def test_repr(self):
        rcs = RotatedGeogCS(
            30, 40, north_pole_grid_longitude=50, ellipsoid=GeogCS(6371229)
        )
        expected = (
            "RotatedGeogCS(30.0, 40.0, "
            "north_pole_grid_longitude=50.0, ellipsoid=GeogCS(6371229.0))"
        )
        self.assertEqual(expected, repr(rcs))

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        expected = "RotatedGeogCS(30.0, 40.0, north_pole_grid_longitude=50.0)"
        self.assertEqual(expected, repr(rcs))

        rcs = RotatedGeogCS(30, 40)
        expected = "RotatedGeogCS(30.0, 40.0)"
        self.assertEqual(expected, repr(rcs))


class Test_RotatedGeogCS_str(tests.IrisTest):
    def test_str(self):
        rcs = RotatedGeogCS(
            30, 40, north_pole_grid_longitude=50, ellipsoid=GeogCS(6371229)
        )
        expected = (
            "RotatedGeogCS(30.0, 40.0, "
            "north_pole_grid_longitude=50.0, ellipsoid=GeogCS(6371229.0))"
        )
        self.assertEqual(expected, str(rcs))

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        expected = "RotatedGeogCS(30.0, 40.0, north_pole_grid_longitude=50.0)"
        self.assertEqual(expected, str(rcs))

        rcs = RotatedGeogCS(30, 40)
        expected = "RotatedGeogCS(30.0, 40.0)"
        self.assertEqual(expected, str(rcs))


class Test_TransverseMercator_construction(tests.IrisTest):
    def test_osgb(self):
        tm = osgb()
        self.assertXMLElement(
            tm, ("coord_systems", "TransverseMercator_osgb.xml")
        )


class Test_TransverseMercator_repr(tests.IrisTest):
    def test_osgb(self):
        tm = osgb()
        expected = (
            "TransverseMercator(latitude_of_projection_origin=49.0, longitude_of_central_meridian=-2.0, "
            "false_easting=-400.0, false_northing=100.0, scale_factor_at_central_meridian=0.9996012717, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909))"
        )
        self.assertEqual(expected, repr(tm))


class Test_TransverseMercator_as_cartopy_crs(tests.IrisTest):
    def test_as_cartopy_crs(self):
        latitude_of_projection_origin = 49.0
        longitude_of_central_meridian = -2.0
        false_easting = -40000.0
        false_northing = 10000.0
        scale_factor_at_central_meridian = 0.9996012717
        ellipsoid = GeogCS(
            semi_major_axis=6377563.396, semi_minor_axis=6356256.909
        )

        tmerc_cs = TransverseMercator(
            latitude_of_projection_origin,
            longitude_of_central_meridian,
            false_easting,
            false_northing,
            scale_factor_at_central_meridian,
            ellipsoid=ellipsoid,
        )

        expected = ccrs.TransverseMercator(
            central_longitude=longitude_of_central_meridian,
            central_latitude=latitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor=scale_factor_at_central_meridian,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
        )

        res = tmerc_cs.as_cartopy_crs()
        self.assertEqual(res, expected)


class Test_TransverseMercator_as_cartopy_projection(tests.IrisTest):
    def test_as_cartopy_projection(self):
        latitude_of_projection_origin = 49.0
        longitude_of_central_meridian = -2.0
        false_easting = -40000.0
        false_northing = 10000.0
        scale_factor_at_central_meridian = 0.9996012717
        ellipsoid = GeogCS(
            semi_major_axis=6377563.396, semi_minor_axis=6356256.909
        )

        tmerc_cs = TransverseMercator(
            latitude_of_projection_origin,
            longitude_of_central_meridian,
            false_easting,
            false_northing,
            scale_factor_at_central_meridian,
            ellipsoid=ellipsoid,
        )

        expected = ccrs.TransverseMercator(
            central_longitude=longitude_of_central_meridian,
            central_latitude=latitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor=scale_factor_at_central_meridian,
            globe=ccrs.Globe(
                semimajor_axis=6377563.396,
                semiminor_axis=6356256.909,
                ellipse=None,
            ),
        )

        res = tmerc_cs.as_cartopy_projection()
        self.assertEqual(res, expected)


class Test_Stereographic_construction(tests.IrisTest):
    def test_stereo(self):
        st = stereo()
        self.assertXMLElement(st, ("coord_systems", "Stereographic.xml"))


class Test_Stereographic_repr(tests.IrisTest):
    def test_stereo(self):
        st = stereo()
        expected = (
            "Stereographic(central_lat=-90.0, central_lon=-45.0, "
            "false_easting=100.0, false_northing=200.0, true_scale_lat=None, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909))"
        )
        self.assertEqual(expected, repr(st))


class Test_Stereographic_as_cartopy_crs(tests.IrisTest):
    def test_as_cartopy_crs(self):
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

        res = st.as_cartopy_crs()
        self.assertEqual(res, expected)


class Test_Stereographic_as_cartopy_projection(tests.IrisTest):
    def test_as_cartopy_projection(self):
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

        res = st.as_cartopy_projection()
        self.assertEqual(res, expected)


class Test_LambertConformal(tests.GraphicsTest):
    def test_fail_secant_latitudes_none(self):
        emsg = "secant latitudes"
        with self.assertRaisesRegex(ValueError, emsg):
            LambertConformal(secant_latitudes=())

    def test_fail_secant_latitudes_excessive(self):
        emsg = "secant latitudes"
        with self.assertRaisesRegex(ValueError, emsg):
            LambertConformal(secant_latitudes=(1, 2, 3))

    def test_secant_latitudes_single_value(self):
        lat_1 = 40
        lcc = LambertConformal(secant_latitudes=lat_1)
        ccrs = lcc.as_cartopy_crs()
        self.assertEqual(lat_1, ccrs.proj4_params["lat_1"])
        self.assertNotIn("lat_2", ccrs.proj4_params)

    def test_secant_latitudes(self):
        lat_1, lat_2 = 40, 41
        lcc = LambertConformal(secant_latitudes=(lat_1, lat_2))
        ccrs = lcc.as_cartopy_crs()
        self.assertEqual(lat_1, ccrs.proj4_params["lat_1"])
        self.assertEqual(lat_2, ccrs.proj4_params["lat_2"])

    def test_north_cutoff(self):
        lcc = LambertConformal(0, 0, secant_latitudes=(30, 60))
        ccrs = lcc.as_cartopy_crs()
        self.assertEqual(ccrs.cutoff, -30)

    def test_south_cutoff(self):
        lcc = LambertConformal(0, 0, secant_latitudes=(-30, -60))
        ccrs = lcc.as_cartopy_crs()
        self.assertEqual(ccrs.cutoff, 30)


if __name__ == "__main__":
    tests.main()
