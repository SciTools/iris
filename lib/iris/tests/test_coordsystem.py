# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import cartopy.crs as ccrs
import pytest

from iris.coord_systems import (
    GeogCS,
    LambertConformal,
    RotatedGeogCS,
    TransverseMercator,
)
import iris.coords
import iris.cube
from iris.tests import _shared_utils
import iris.tests.stock
from iris.warnings import IrisUserWarning


def osgb():
    return TransverseMercator(
        latitude_of_projection_origin=49,
        longitude_of_central_meridian=-2,
        false_easting=-400,
        false_northing=100,
        scale_factor_at_central_meridian=0.9996012717,
        ellipsoid=GeogCS(6377563.396, 6356256.909),
    )


class TestCoordSystemLookup:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.lat_lon_cube()

    def test_hit_name(self):
        assert isinstance(self.cube.coord_system("GeogCS"), GeogCS)

    def test_hit_type(self):
        assert isinstance(self.cube.coord_system(GeogCS), GeogCS)

    def test_miss(self):
        assert self.cube.coord_system(RotatedGeogCS) is None

    def test_empty(self):
        assert isinstance(self.cube.coord_system(GeogCS), GeogCS)
        assert self.cube.coord_system(None) is not None
        assert isinstance(self.cube.coord_system(None), GeogCS)
        assert self.cube.coord_system() is not None
        assert isinstance(self.cube.coord_system(), GeogCS)

        for coord in self.cube.coords():
            coord.coord_system = None

        assert self.cube.coord_system(GeogCS) is None
        assert self.cube.coord_system(None) is None
        assert self.cube.coord_system() is None


class TestCoordSystemSame:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cs1 = iris.coord_systems.GeogCS(6371229)
        self.cs2 = iris.coord_systems.GeogCS(6371229)
        self.cs3 = iris.coord_systems.RotatedGeogCS(30, 30, ellipsoid=GeogCS(6371229))

    def test_simple(self):
        a = self.cs1
        b = self.cs2
        assert a == b

    def test_different_class(self):
        a = self.cs1
        b = self.cs3
        assert a != b

    def test_different_public_attributes(self):
        a = self.cs1
        b = self.cs2
        a.foo = "a"

        # check that that attribute was added (just in case)
        assert a.foo == "a"

        # a and b should not be the same
        assert a != b

        # a and b should be the same
        b.foo = "a"
        assert a == b

        b.foo = "b"
        # a and b should not be the same
        assert a != b


class Test_CoordSystem_xml_element:
    def test_rotated(self):
        cs = RotatedGeogCS(30, 40, ellipsoid=GeogCS(6371229))
        _shared_utils.assert_XML_element(
            cs, ("coord_systems", "CoordSystem_xml_element.xml")
        )


class Test_GeogCS_construction:
    # Test Ellipsoid constructor
    # Don't care about testing the units, it has no logic specific to this class.

    def test_sphere_param(self):
        cs = GeogCS(6543210)
        _shared_utils.assert_XML_element(
            cs, ("coord_systems", "GeogCS_init_sphere.xml")
        )

    def test_no_major(self):
        cs = GeogCS(semi_minor_axis=6500000, inverse_flattening=151.42814163388104)
        _shared_utils.assert_XML_element(
            cs, ("coord_systems", "GeogCS_init_no_major.xml")
        )

    def test_no_minor(self):
        cs = GeogCS(semi_major_axis=6543210, inverse_flattening=151.42814163388104)
        _shared_utils.assert_XML_element(
            cs, ("coord_systems", "GeogCS_init_no_minor.xml")
        )

    def test_no_invf(self):
        cs = GeogCS(semi_major_axis=6543210, semi_minor_axis=6500000)
        _shared_utils.assert_XML_element(
            cs, ("coord_systems", "GeogCS_init_no_invf.xml")
        )

    def test_invalid_ellipsoid_params(self):
        # no params
        with pytest.raises(ValueError, match="No ellipsoid specified"):
            GeogCS()

        # over specified
        with pytest.raises(ValueError, match="Ellipsoid is overspecified"):
            GeogCS(6543210, 6500000, 151.42814163388104)

        # under specified
        with pytest.raises(ValueError, match="Insufficient ellipsoid specification"):
            GeogCS(None, 6500000, None)
        with pytest.raises(ValueError, match="Insufficient ellipsoid specification"):
            GeogCS(None, None, 151.42814163388104)


class Test_GeogCS_repr:
    def test_repr(self):
        cs = GeogCS(6543210, 6500000)
        expected = "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0)"
        assert expected == repr(cs)


class Test_GeogCS_str:
    def test_str(self):
        cs = GeogCS(6543210, 6500000)
        expected = "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0)"
        assert expected == str(cs)


class Test_GeogCS_as_cartopy_globe:
    def test_as_cartopy_globe(self):
        cs = GeogCS(6543210, 6500000)
        # Can't check equality directly, so use the proj4 params instead.
        res = cs.as_cartopy_globe().to_proj4_params()
        expected = {"a": 6543210, "b": 6500000}
        assert res == expected


class Test_GeogCS_as_cartopy_projection:
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

        assert res == expected


class Test_GeogCS_as_cartopy_crs:
    def test_as_cartopy_crs(self):
        cs = GeogCS(6543210, 6500000)
        res = cs.as_cartopy_crs()
        globe = ccrs.Globe(
            semimajor_axis=6543210.0,
            semiminor_axis=6500000.0,
            ellipse=None,
        )
        expected = ccrs.Geodetic(globe)
        assert res == expected


class Test_GeogCS_equality:
    """Test cached values don't break GeogCS equality."""

    def test_as_cartopy_globe(self):
        cs_const = GeogCS(6543210, 6500000)
        cs_mut = GeogCS(6543210, 6500000)
        initial_globe = cs_mut.as_cartopy_globe()
        new_globe = cs_mut.as_cartopy_globe()

        assert new_globe is initial_globe
        assert cs_const == cs_mut

    def test_as_cartopy_projection(self):
        cs_const = GeogCS(6543210, 6500000)
        cs_mut = GeogCS(6543210, 6500000)
        initial_projection = cs_mut.as_cartopy_projection()
        initial_globe = initial_projection.globe
        new_projection = cs_mut.as_cartopy_projection()
        new_globe = new_projection.globe

        assert new_globe is initial_globe
        assert cs_const == cs_mut

    def test_as_cartopy_crs(self):
        cs_const = GeogCS(6543210, 6500000)
        cs_mut = GeogCS(6543210, 6500000)
        initial_crs = cs_mut.as_cartopy_crs()
        initial_globe = initial_crs.globe
        new_crs = cs_mut.as_cartopy_crs()
        new_globe = new_crs.globe

        assert new_crs is initial_crs
        assert new_globe is initial_globe
        assert cs_const == cs_mut

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

        assert cs_const.semi_major_axis == 6500000
        assert cs_mut.semi_major_axis == 6500000
        assert cs_const == cs_mut


class Test_GeogCS_mutation:
    """Test that altering attributes of a GeogCS instance behaves as expected."""

    def test_semi_major_axis_change(self):
        # Clear datum
        # Clear caches
        cs = GeogCS.from_datum("OSGB 1936")
        _ = cs.as_cartopy_crs()
        assert cs.datum == "OSGB 1936"
        cs.semi_major_axis = 6000000
        assert cs.datum is None
        assert cs.as_cartopy_globe().semimajor_axis == 6000000

    def test_semi_major_axis_no_change(self):
        # Datum untouched
        # Caches untouched
        cs = GeogCS.from_datum("OSGB 1936")
        initial_crs = cs.as_cartopy_crs()
        assert cs.datum == "OSGB 1936"
        cs.semi_major_axis = 6377563.396
        assert cs.datum == "OSGB 1936"
        new_crs = cs.as_cartopy_crs()
        assert new_crs is initial_crs

    def test_semi_minor_axis_change(self):
        # Clear datum
        # Clear caches
        cs = GeogCS.from_datum("OSGB 1936")
        _ = cs.as_cartopy_crs()
        assert cs.datum == "OSGB 1936"
        cs.semi_minor_axis = 6000000
        assert cs.datum is None
        assert cs.as_cartopy_globe().semiminor_axis == 6000000

    def test_semi_minor_axis_no_change(self):
        # Datum untouched
        # Caches untouched
        cs = GeogCS.from_datum("OSGB 1936")
        initial_crs = cs.as_cartopy_crs()
        assert cs.datum == "OSGB 1936"
        cs.semi_minor_axis = 6356256.909237285
        assert cs.datum == "OSGB 1936"
        new_crs = cs.as_cartopy_crs()
        assert new_crs is initial_crs

    def test_datum_change(self):
        # Semi-major axis changes
        # All internal ellipoid values set to None
        # CRS changes
        cs = GeogCS(6543210, 6500000)
        _ = cs.as_cartopy_crs()
        assert "_globe" in cs.__dict__
        assert "_crs" in cs.__dict__
        assert cs.semi_major_axis == 6543210
        cs.datum = "OSGB 1936"
        assert cs.as_cartopy_crs().datum == "OSGB 1936"
        assert cs.__dict__["_semi_major_axis"] is None
        assert cs.__dict__["_semi_minor_axis"] is None
        assert cs.__dict__["_inverse_flattening"] is None
        assert cs.semi_major_axis == 6377563.396

    def test_datum_no_change(self):
        # Caches untouched
        cs = GeogCS.from_datum("OSGB 1936")
        initial_crs = cs.as_cartopy_crs()
        cs.datum = "OSGB 1936"
        new_crs = cs.as_cartopy_crs()
        assert new_crs is initial_crs

    def test_inverse_flattening_change(self):
        # Caches untouched
        # Axes unchanged (this behaviour is odd, but matches existing behaviour)
        # Warning about lack of effect on other aspects
        cs = GeogCS(6543210, 6500000)
        initial_crs = cs.as_cartopy_crs()
        with pytest.warns(
            IrisUserWarning,
            match="Setting inverse_flattening does not affect other properties of the GeogCS object.",
        ):
            cs.inverse_flattening = cs.inverse_flattening + 1
        new_crs = cs.as_cartopy_crs()
        assert new_crs is initial_crs
        assert cs.semi_major_axis == 6543210
        assert cs.semi_minor_axis == 6500000


class Test_RotatedGeogCS_construction:
    def test_init(self):
        rcs = RotatedGeogCS(
            30, 40, north_pole_grid_longitude=50, ellipsoid=GeogCS(6371229)
        )
        _shared_utils.assert_XML_element(
            rcs, ("coord_systems", "RotatedGeogCS_init.xml")
        )

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        _shared_utils.assert_XML_element(
            rcs, ("coord_systems", "RotatedGeogCS_init_a.xml")
        )

        rcs = RotatedGeogCS(30, 40)
        _shared_utils.assert_XML_element(
            rcs, ("coord_systems", "RotatedGeogCS_init_b.xml")
        )


class Test_RotatedGeogCS_repr:
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
        assert expected == repr(rcs)

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        expected = "RotatedGeogCS(30.0, 40.0, north_pole_grid_longitude=50.0)"
        assert expected == repr(rcs)

        rcs = RotatedGeogCS(30, 40)
        expected = "RotatedGeogCS(30.0, 40.0)"
        assert expected == repr(rcs)


class Test_RotatedGeogCS_str:
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
        assert expected == str(rcs)

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        expected = "RotatedGeogCS(30.0, 40.0, north_pole_grid_longitude=50.0)"
        assert expected == str(rcs)

        rcs = RotatedGeogCS(30, 40)
        expected = "RotatedGeogCS(30.0, 40.0)"
        assert expected == str(rcs)


class Test_TransverseMercator_construction:
    def test_osgb(self):
        tm = osgb()
        _shared_utils.assert_XML_element(
            tm, ("coord_systems", "TransverseMercator_osgb.xml")
        )


class Test_TransverseMercator_repr:
    def test_osgb(self):
        tm = osgb()
        expected = (
            "TransverseMercator(latitude_of_projection_origin=49.0, longitude_of_central_meridian=-2.0, "
            "false_easting=-400.0, false_northing=100.0, scale_factor_at_central_meridian=0.9996012717, "
            "ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909))"
        )
        assert expected == repr(tm)


class Test_TransverseMercator_as_cartopy_crs:
    def test_as_cartopy_crs(self):
        latitude_of_projection_origin = 49.0
        longitude_of_central_meridian = -2.0
        false_easting = -40000.0
        false_northing = 10000.0
        scale_factor_at_central_meridian = 0.9996012717
        ellipsoid = GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909)

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
        assert res == expected


class Test_TransverseMercator_as_cartopy_projection:
    def test_as_cartopy_projection(self):
        latitude_of_projection_origin = 49.0
        longitude_of_central_meridian = -2.0
        false_easting = -40000.0
        false_northing = 10000.0
        scale_factor_at_central_meridian = 0.9996012717
        ellipsoid = GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909)

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
        assert res == expected


class Test_LambertConformal(_shared_utils.GraphicsTest):
    def test_fail_secant_latitudes_none(self):
        emsg = "secant latitudes"
        with pytest.raises(ValueError, match=emsg):
            LambertConformal(secant_latitudes=())

    def test_fail_secant_latitudes_excessive(self):
        emsg = "secant latitudes"
        with pytest.raises(ValueError, match=emsg):
            LambertConformal(secant_latitudes=(1, 2, 3))

    def test_secant_latitudes_single_value(self):
        lat_1 = 40
        lcc = LambertConformal(secant_latitudes=lat_1)
        ccrs = lcc.as_cartopy_crs()
        assert lat_1 == ccrs.proj4_params["lat_1"]
        assert "lat_2" not in ccrs.proj4_params

    def test_secant_latitudes(self):
        lat_1, lat_2 = 40, 41
        lcc = LambertConformal(secant_latitudes=(lat_1, lat_2))
        ccrs = lcc.as_cartopy_crs()
        assert lat_1 == ccrs.proj4_params["lat_1"]
        assert lat_2 == ccrs.proj4_params["lat_2"]

    def test_north_cutoff(self):
        lcc = LambertConformal(0, 0, secant_latitudes=(30, 60))
        ccrs = lcc.as_cartopy_crs()
        assert ccrs.cutoff == -30

    def test_south_cutoff(self):
        lcc = LambertConformal(0, 0, secant_latitudes=(-30, -60))
        ccrs = lcc.as_cartopy_crs()
        assert ccrs.cutoff == 30


class Test_Datums:
    def test_default_none(self):
        cs = GeogCS(6543210, 6500000)  # Arbitrary radii
        cartopy_crs = cs.as_cartopy_crs()
        assert cartopy_crs.datum.name == "unknown"

    def test_set_persist(self):
        cs = GeogCS.from_datum(datum="WGS84")
        cartopy_crs = cs.as_cartopy_crs()
        assert cartopy_crs.datum.name == "World Geodetic System 1984"

        cs = GeogCS.from_datum(datum="OSGB36")
        cartopy_crs = cs.as_cartopy_crs()
        assert cartopy_crs.datum.name == "OSGB 1936"
