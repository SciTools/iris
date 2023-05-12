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
        self.assertNotEqual(a, b)

    def test_different_public_attributes(self):
        a = self.cs1
        b = self.cs2
        a.foo = "a"

        # check that that attribute was added (just in case)
        self.assertEqual(a.foo, "a")

        # a and b should not be the same
        self.assertNotEqual(a, b)

        # a and b should be the same
        b.foo = "a"
        self.assertEqual(a, b)

        b.foo = "b"
        # a and b should not be the same
        self.assertNotEqual(a, b)


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


class Test_GeogCS_as_cartopy_projection(tests.IrisTest):
    def test_as_cartopy_projection(self):
        geogcs_args = {
            "semi_major_axis": 6543210,
            "semi_minor_axis": 6500000,
            "longitude_of_prime_meridian": 30,
        }
        cs = GeogCS(**geogcs_args)
        res = cs.as_cartopy_projection()

        globe = ccrs.Globe(
            semimajor_axis=geogcs_args["semi_major_axis"],
            semiminor_axis=geogcs_args["semi_minor_axis"],
            ellipse=None,
        )
        expected = ccrs.PlateCarree(
            globe=globe,
            central_longitude=geogcs_args["longitude_of_prime_meridian"],
        )

        self.assertEqual(res, expected)


class Test_GeogCS_as_cartopy_crs(tests.IrisTest):
    def test_as_cartopy_crs(self):
        cs = GeogCS(6543210, 6500000)
        res = cs.as_cartopy_crs()
        globe = ccrs.Globe(
            semimajor_axis=6543210.0,
            semiminor_axis=6500000.0,
            ellipse=None,
        )
        expected = ccrs.Geodetic(globe)
        self.assertEqual(res, expected)


class Test_GeogCS_equality(tests.IrisTest):
    """Test cached values don't break GeogCS equality"""

    def test_as_cartopy_globe(self):
        cs_const = GeogCS(6543210, 6500000)
        cs_mut = GeogCS(6543210, 6500000)
        initial_globe = cs_mut.as_cartopy_globe()
        new_globe = cs_mut.as_cartopy_globe()

        self.assertIs(new_globe, initial_globe)
        self.assertEqual(cs_const, cs_mut)

    def test_as_cartopy_projection(self):
        cs_const = GeogCS(6543210, 6500000)
        cs_mut = GeogCS(6543210, 6500000)
        initial_projection = cs_mut.as_cartopy_projection()
        initial_globe = initial_projection.globe
        new_projection = cs_mut.as_cartopy_projection()
        new_globe = new_projection.globe

        self.assertIs(new_globe, initial_globe)
        self.assertEqual(cs_const, cs_mut)

    def test_as_cartopy_crs(self):
        cs_const = GeogCS(6543210, 6500000)
        cs_mut = GeogCS(6543210, 6500000)
        initial_crs = cs_mut.as_cartopy_crs()
        initial_globe = initial_crs.globe
        new_crs = cs_mut.as_cartopy_crs()
        new_globe = new_crs.globe

        self.assertIs(new_crs, initial_crs)
        self.assertIs(new_globe, initial_globe)
        self.assertEqual(cs_const, cs_mut)

    def test_update_to_equivalent(self):
        cs_const = GeogCS(6500000, 6000000)
        # Cause caching
        _ = cs_const.as_cartopy_crs()

        cs_mut = GeogCS(6543210, 6000000)
        # Cause caching
        _ = cs_mut.as_cartopy_crs()
        # Set value
        cs_mut.semi_major_axis = 6500000
        cs_mut.inverse_flattening = 13

        self.assertEqual(cs_const.semi_major_axis, 6500000)
        self.assertEqual(cs_mut.semi_major_axis, 6500000)
        self.assertEqual(cs_const, cs_mut)


class Test_GeogCS_mutation(tests.IrisTest):
    "Test that altering attributes of a GeogCS instance behaves as expected"

    def test_semi_major_axis_change(self):
        # Clear datum
        # Clear caches
        cs = GeogCS.from_datum("OSGB 1936")
        _ = cs.as_cartopy_crs()
        self.assertEqual(cs.datum, "OSGB 1936")
        cs.semi_major_axis = 6000000
        self.assertIsNone(cs.datum)
        self.assertEqual(cs.as_cartopy_globe().semimajor_axis, 6000000)

    def test_semi_major_axis_no_change(self):
        # Datum untouched
        # Caches untouched
        cs = GeogCS.from_datum("OSGB 1936")
        initial_crs = cs.as_cartopy_crs()
        self.assertEqual(cs.datum, "OSGB 1936")
        cs.semi_major_axis = 6377563.396
        self.assertEqual(cs.datum, "OSGB 1936")
        new_crs = cs.as_cartopy_crs()
        self.assertIs(new_crs, initial_crs)

    def test_semi_minor_axis_change(self):
        # Clear datum
        # Clear caches
        cs = GeogCS.from_datum("OSGB 1936")
        _ = cs.as_cartopy_crs()
        self.assertEqual(cs.datum, "OSGB 1936")
        cs.semi_minor_axis = 6000000
        self.assertIsNone(cs.datum)
        self.assertEqual(cs.as_cartopy_globe().semiminor_axis, 6000000)

    def test_semi_minor_axis_no_change(self):
        # Datum untouched
        # Caches untouched
        cs = GeogCS.from_datum("OSGB 1936")
        initial_crs = cs.as_cartopy_crs()
        self.assertEqual(cs.datum, "OSGB 1936")
        cs.semi_minor_axis = 6356256.909237285
        self.assertEqual(cs.datum, "OSGB 1936")
        new_crs = cs.as_cartopy_crs()
        self.assertIs(new_crs, initial_crs)

    def test_datum_change(self):
        # Semi-major axis changes
        # All internal ellipoid values set to None
        # CRS changes
        cs = GeogCS(6543210, 6500000)
        _ = cs.as_cartopy_crs()
        self.assertTrue("_globe" in cs.__dict__)
        self.assertTrue("_crs" in cs.__dict__)
        self.assertEqual(cs.semi_major_axis, 6543210)
        cs.datum = "OSGB 1936"
        self.assertEqual(cs.as_cartopy_crs().datum, "OSGB 1936")
        self.assertIsNone(cs.__dict__["_semi_major_axis"])
        self.assertIsNone(cs.__dict__["_semi_minor_axis"])
        self.assertIsNone(cs.__dict__["_inverse_flattening"])
        self.assertEqual(cs.semi_major_axis, 6377563.396)

    def test_datum_no_change(self):
        # Caches untouched
        cs = GeogCS.from_datum("OSGB 1936")
        initial_crs = cs.as_cartopy_crs()
        cs.datum = "OSGB 1936"
        new_crs = cs.as_cartopy_crs()
        self.assertIs(new_crs, initial_crs)

    def test_inverse_flattening_change(self):
        # Caches untouched
        # Axes unchanged (this behaviour is odd, but matches existing behaviour)
        # Warning about lack of effect on other aspects
        cs = GeogCS(6543210, 6500000)
        initial_crs = cs.as_cartopy_crs()
        with self.assertWarnsRegex(
            UserWarning,
            "Setting inverse_flattening does not affect other properties of the GeogCS object.",
        ):
            cs.inverse_flattening = cs.inverse_flattening + 1
        new_crs = cs.as_cartopy_crs()
        self.assertIs(new_crs, initial_crs)
        self.assertEqual(cs.semi_major_axis, 6543210)
        self.assertEqual(cs.semi_minor_axis, 6500000)


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
            30,
            40,
            north_pole_grid_longitude=50,
            ellipsoid=GeogCS(6371229),
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
            30,
            40,
            north_pole_grid_longitude=50,
            ellipsoid=GeogCS(6371229),
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


class Test_Datums(tests.IrisTest):
    def test_default_none(self):
        cs = GeogCS(6543210, 6500000)  # Arbitrary radii
        cartopy_crs = cs.as_cartopy_crs()
        self.assertMultiLineEqual(cartopy_crs.datum.name, "unknown")

    def test_set_persist(self):
        cs = GeogCS.from_datum(datum="WGS84")
        cartopy_crs = cs.as_cartopy_crs()
        self.assertMultiLineEqual(
            cartopy_crs.datum.name, "World Geodetic System 1984"
        )

        cs = GeogCS.from_datum(datum="OSGB36")
        cartopy_crs = cs.as_cartopy_crs()
        self.assertMultiLineEqual(cartopy_crs.datum.name, "OSGB 1936")


if __name__ == "__main__":
    tests.main()
