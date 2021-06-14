# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import unittest

import numpy as np

import iris
import iris.analysis.calculus
import iris.coord_systems
import iris.coords
from iris.coords import DimCoord
import iris.cube
import iris.tests.stock


class TestCubeDelta(tests.IrisTest):
    @tests.skip_data
    def test_invalid(self):
        cube = iris.tests.stock.realistic_4d()
        with self.assertRaises(iris.exceptions.CoordinateMultiDimError):
            _ = iris.analysis.calculus.cube_delta(cube, "surface_altitude")
        with self.assertRaises(iris.exceptions.CoordinateMultiDimError):
            _ = iris.analysis.calculus.cube_delta(cube, "altitude")
        with self.assertRaises(ValueError):
            _ = iris.analysis.calculus.cube_delta(cube, "forecast_period")

    def test_delta_coord_lookup(self):
        cube = iris.cube.Cube(np.arange(10), standard_name="air_temperature")
        # Add a coordinate with a lot of metadata.
        coord = iris.coords.DimCoord(
            np.arange(10),
            long_name="projection_x_coordinate",
            var_name="foo",
            attributes={"source": "testing"},
            units="m",
            coord_system=iris.coord_systems.OSGB(),
        )
        cube.add_dim_coord(coord, 0)
        delta = iris.analysis.calculus.cube_delta(
            cube, "projection_x_coordinate"
        )
        delta_coord = delta.coord("projection_x_coordinate")
        self.assertEqual(delta_coord, delta.coord(coord))
        self.assertEqual(coord, cube.coord(delta_coord))


class TestDeltaAndMidpoint(tests.IrisTest):
    def _simple_filename(self, suffix):
        return tests.get_result_path(
            ("analysis", "delta_and_midpoint", "simple%s.cml" % suffix)
        )

    def test_simple1_delta_midpoint(self):
        a = iris.coords.DimCoord(
            (np.arange(4, dtype=np.float32) * 90) - 180,
            long_name="foo",
            units="degrees",
            circular=True,
        )
        self.assertXMLElement(a, self._simple_filename("1"))

        delta = iris.analysis.calculus._construct_delta_coord(a)
        self.assertXMLElement(delta, self._simple_filename("1_delta"))

        midpoint = iris.analysis.calculus._construct_midpoint_coord(a)
        self.assertXMLElement(midpoint, self._simple_filename("1_midpoint"))

    def test_simple2_delta_midpoint(self):
        a = iris.coords.DimCoord(
            (np.arange(4, dtype=np.float32) * -90) + 180,
            long_name="foo",
            units="degrees",
            circular=True,
        )
        self.assertXMLElement(a, self._simple_filename("2"))

        delta = iris.analysis.calculus._construct_delta_coord(a)
        self.assertXMLElement(delta, self._simple_filename("2_delta"))

        midpoint = iris.analysis.calculus._construct_midpoint_coord(a)
        self.assertXMLElement(midpoint, self._simple_filename("2_midpoint"))

    def test_simple3_delta_midpoint(self):
        a = iris.coords.DimCoord(
            (np.arange(4, dtype=np.float32) * 90) - 180,
            long_name="foo",
            units="degrees",
            circular=True,
        )
        a.guess_bounds(0.5)
        self.assertXMLElement(a, self._simple_filename("3"))

        delta = iris.analysis.calculus._construct_delta_coord(a)
        self.assertXMLElement(delta, self._simple_filename("3_delta"))

        midpoint = iris.analysis.calculus._construct_midpoint_coord(a)
        self.assertXMLElement(midpoint, self._simple_filename("3_midpoint"))

    def test_simple4_delta_midpoint(self):
        a = iris.coords.AuxCoord(
            np.arange(4, dtype=np.float32) * 90 - 180,
            long_name="foo",
            units="degrees",
        )
        a.guess_bounds()
        b = a.copy()
        self.assertXMLElement(b, self._simple_filename("4"))

        delta = iris.analysis.calculus._construct_delta_coord(b)
        self.assertXMLElement(delta, self._simple_filename("4_delta"))

        midpoint = iris.analysis.calculus._construct_midpoint_coord(b)
        self.assertXMLElement(midpoint, self._simple_filename("4_midpoint"))

    def test_simple5_not_degrees_delta_midpoint(self):
        # Not sure it makes sense to have a circular coordinate which does not have a modulus but test it anyway.
        a = iris.coords.DimCoord(
            np.arange(4, dtype=np.float32) * 90 - 180,
            long_name="foo",
            units="meter",
            circular=True,
        )
        self.assertXMLElement(a, self._simple_filename("5"))

        delta = iris.analysis.calculus._construct_delta_coord(a)
        self.assertXMLElement(delta, self._simple_filename("5_delta"))

        midpoints = iris.analysis.calculus._construct_midpoint_coord(a)
        self.assertXMLElement(midpoints, self._simple_filename("5_midpoint"))

    def test_simple6_delta_midpoint(self):
        a = iris.coords.DimCoord(
            np.arange(5, dtype=np.float32),
            long_name="foo",
            units="count",
            circular=True,
        )
        midpoints = iris.analysis.calculus._construct_midpoint_coord(a)
        self.assertXMLElement(midpoints, self._simple_filename("6"))

    def test_singular_delta(self):
        # Test single valued coordinate mid-points when circular
        lon = iris.coords.DimCoord(
            np.float32(-180.0), "latitude", units="degrees", circular=True
        )

        r_expl = iris.analysis.calculus._construct_delta_coord(lon)
        self.assertXMLElement(
            r_expl,
            (
                "analysis",
                "delta_and_midpoint",
                "delta_one_element_explicit.xml",
            ),
        )

        # Test single valued coordinate mid-points when not circular
        lon.circular = False
        with self.assertRaises(ValueError):
            iris.analysis.calculus._construct_delta_coord(lon)

    def test_singular_midpoint(self):
        # Test single valued coordinate mid-points when circular
        lon = iris.coords.DimCoord(
            np.float32(-180.0), "latitude", units="degrees", circular=True
        )

        r_expl = iris.analysis.calculus._construct_midpoint_coord(lon)
        self.assertXMLElement(
            r_expl,
            (
                "analysis",
                "delta_and_midpoint",
                "midpoint_one_element_explicit.xml",
            ),
        )

        # Test single valued coordinate mid-points when not circular
        lon.circular = False
        with self.assertRaises(ValueError):
            iris.analysis.calculus._construct_midpoint_coord(lon)


class TestCoordTrig(tests.IrisTest):
    def setUp(self):
        points = np.arange(20, dtype=np.float32) * 2.3
        bounds = np.concatenate([[points - 0.5 * 2.3], [points + 0.5 * 2.3]]).T
        self.lat = iris.coords.AuxCoord(
            points, "latitude", units="degrees", bounds=bounds
        )
        self.rlat = iris.coords.AuxCoord(
            np.deg2rad(points),
            "latitude",
            units="radians",
            bounds=np.deg2rad(bounds),
        )

    def test_sin(self):
        sin_of_coord = iris.analysis.calculus._coord_sin(self.lat)
        sin_of_coord_radians = iris.analysis.calculus._coord_sin(self.rlat)

        # Check the values are correct (within a tolerance)
        np.testing.assert_array_almost_equal(
            np.sin(self.rlat.points), sin_of_coord.points
        )
        np.testing.assert_array_almost_equal(
            np.sin(self.rlat.bounds), sin_of_coord.bounds
        )

        # Check that the results of the sin function are almost equal when operating on a coord with degrees and radians
        np.testing.assert_array_almost_equal(
            sin_of_coord.points, sin_of_coord_radians.points
        )
        np.testing.assert_array_almost_equal(
            sin_of_coord.bounds, sin_of_coord_radians.bounds
        )

        self.assertEqual(sin_of_coord.name(), "sin(latitude)")
        self.assertEqual(sin_of_coord.units, "1")

    def test_cos(self):
        cos_of_coord = iris.analysis.calculus._coord_cos(self.lat)
        cos_of_coord_radians = iris.analysis.calculus._coord_cos(self.rlat)

        # Check the values are correct (within a tolerance)
        np.testing.assert_array_almost_equal(
            np.cos(self.rlat.points), cos_of_coord.points
        )
        np.testing.assert_array_almost_equal(
            np.cos(self.rlat.bounds), cos_of_coord.bounds
        )

        # Check that the results of the cos function are almost equal when operating on a coord with degrees and radians
        np.testing.assert_array_almost_equal(
            cos_of_coord.points, cos_of_coord_radians.points
        )
        np.testing.assert_array_almost_equal(
            cos_of_coord.bounds, cos_of_coord_radians.bounds
        )

        # Now that we have tested the points & bounds, remove them and just test the xml
        cos_of_coord = cos_of_coord.copy(
            points=np.array([1], dtype=np.float32)
        )
        cos_of_coord_radians = cos_of_coord_radians.copy(
            points=np.array([1], dtype=np.float32)
        )

        self.assertXMLElement(
            cos_of_coord, ("analysis", "calculus", "cos_simple.xml")
        )
        self.assertXMLElement(
            cos_of_coord_radians,
            ("analysis", "calculus", "cos_simple_radians.xml"),
        )


class TestCalculusSimple3(tests.IrisTest):
    def setUp(self):
        data = np.arange(2500, dtype=np.float32).reshape(50, 50)
        cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")

        self.lonlat_cs = iris.coord_systems.GeogCS(6371229)
        cube.add_dim_coord(
            DimCoord(
                np.arange(50, dtype=np.float32) * 4.5 - 180,
                "longitude",
                units="degrees",
                coord_system=self.lonlat_cs,
            ),
            0,
        )
        cube.add_dim_coord(
            DimCoord(
                np.arange(50, dtype=np.float32) * 4.5 - 90,
                "latitude",
                units="degrees",
                coord_system=self.lonlat_cs,
            ),
            1,
        )

        self.cube = cube

    def test_diff_wrt_lon(self):
        t = iris.analysis.calculus.differentiate(self.cube, "longitude")

        self.assertCMLApproxData(
            t, ("analysis", "calculus", "handmade2_wrt_lon.cml")
        )

    def test_diff_wrt_lat(self):
        t = iris.analysis.calculus.differentiate(self.cube, "latitude")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "handmade2_wrt_lat.cml")
        )


class TestCalculusSimple2(tests.IrisTest):
    def setUp(self):
        data = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 9],
            ],
            dtype=np.float32,
        )
        cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")

        self.lonlat_cs = iris.coord_systems.GeogCS(6371229)
        cube.add_dim_coord(
            DimCoord(
                np.arange(4, dtype=np.float32) * 90 - 180,
                "longitude",
                units="degrees",
                circular=True,
                coord_system=self.lonlat_cs,
            ),
            0,
        )
        cube.add_dim_coord(
            DimCoord(
                np.arange(5, dtype=np.float32) * 45 - 90,
                "latitude",
                units="degrees",
                coord_system=self.lonlat_cs,
            ),
            1,
        )

        cube.add_aux_coord(
            DimCoord(
                np.arange(4, dtype=np.float32),
                long_name="x",
                units="count",
                circular=True,
            ),
            0,
        )
        cube.add_aux_coord(
            DimCoord(
                np.arange(5, dtype=np.float32), long_name="y", units="count"
            ),
            1,
        )

        self.cube = cube

    def test_diff_wrt_x(self):
        t = iris.analysis.calculus.differentiate(self.cube, "x")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "handmade_wrt_x.cml")
        )

    def test_diff_wrt_y(self):
        t = iris.analysis.calculus.differentiate(self.cube, "y")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "handmade_wrt_y.cml")
        )

    def test_diff_wrt_lon(self):
        t = iris.analysis.calculus.differentiate(self.cube, "longitude")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "handmade_wrt_lon.cml")
        )

    def test_diff_wrt_lat(self):
        t = iris.analysis.calculus.differentiate(self.cube, "latitude")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "handmade_wrt_lat.cml")
        )

    def test_delta_wrt_x(self):
        t = iris.analysis.calculus.cube_delta(self.cube, "x")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "delta_handmade_wrt_x.cml")
        )

    def test_delta_wrt_y(self):
        t = iris.analysis.calculus.cube_delta(self.cube, "y")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "delta_handmade_wrt_y.cml")
        )

    def test_delta_wrt_lon(self):
        t = iris.analysis.calculus.cube_delta(self.cube, "longitude")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "delta_handmade_wrt_lon.cml")
        )

    def test_delta_wrt_lat(self):
        t = iris.analysis.calculus.cube_delta(self.cube, "latitude")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "delta_handmade_wrt_lat.cml")
        )


class TestCalculusSimple1(tests.IrisTest):
    def setUp(self):
        data = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 10],
            ],
            dtype=np.float32,
        )
        cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")

        cube.add_dim_coord(
            DimCoord(
                np.arange(5, dtype=np.float32), long_name="x", units="count"
            ),
            0,
        )
        cube.add_dim_coord(
            DimCoord(
                np.arange(5, dtype=np.float32), long_name="y", units="count"
            ),
            1,
        )

        self.cube = cube

    def test_diff_wrt_x(self):
        t = iris.analysis.calculus.differentiate(self.cube, "x")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "handmade_simple_wrt_x.cml")
        )

    def test_delta_wrt_x(self):
        t = iris.analysis.calculus.cube_delta(self.cube, "x")
        self.assertCMLApproxData(
            t, ("analysis", "calculus", "delta_handmade_simple_wrt_x.cml")
        )


def build_cube(data, spherical=False):
    """
    Create a cube suitable for testing.

    """
    cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")

    nx = data.shape[-1]
    ny = data.shape[-2]
    nz = data.shape[-3] if data.ndim > 2 else None

    dimx = data.ndim - 1
    dimy = data.ndim - 2
    dimz = data.ndim - 3 if data.ndim > 2 else None

    if spherical:
        if spherical == "rotated":
            hcs = iris.coord_systems.RotatedGeogCS(10, 20)
            lon_name, lat_name = "grid_longitude", "grid_latitude"
        else:
            hcs = iris.coord_systems.GeogCS(6321)
            lon_name, lat_name = "longitude", "latitude"
        cube.add_dim_coord(
            DimCoord(
                np.arange(-180, 180, 360.0 / nx, dtype=np.float32),
                lon_name,
                units="degrees",
                coord_system=hcs,
                circular=True,
            ),
            dimx,
        )
        cube.add_dim_coord(
            DimCoord(
                np.arange(-90, 90, 180.0 / ny, dtype=np.float32),
                lat_name,
                units="degrees",
                coord_system=hcs,
            ),
            dimy,
        )

    else:
        cube.add_dim_coord(
            DimCoord(
                np.arange(nx, dtype=np.float32) * 2.21 + 2,
                "projection_x_coordinate",
                units="meters",
            ),
            dimx,
        )
        cube.add_dim_coord(
            DimCoord(
                np.arange(ny, dtype=np.float32) * 25 - 50,
                "projection_y_coordinate",
                units="meters",
            ),
            dimy,
        )

    if nz is None:
        cube.add_aux_coord(
            DimCoord(
                np.array([10], dtype=np.float32),
                long_name="z",
                units="meters",
                attributes={"positive": "up"},
            )
        )
    else:
        cube.add_dim_coord(
            DimCoord(
                np.arange(nz, dtype=np.float32) * 2,
                long_name="z",
                units="meters",
                attributes={"positive": "up"},
            ),
            dimz,
        )

    return cube


class TestCalculusWKnownSolutions(tests.IrisTest):
    def get_coord_pts(self, cube):
        """return (x_pts, x_ones, y_pts, y_ones, z_pts, z_ones) for the given cube."""
        x = cube.coord(axis="X")
        y = cube.coord(axis="Y")
        z = cube.coord(axis="Z")

        if z and z.shape[0] > 1:
            x_shp = (1, 1, x.shape[0])
            y_shp = (1, y.shape[0], 1)
            z_shp = (z.shape[0], 1, 1)
        else:
            x_shp = (1, x.shape[0])
            y_shp = (y.shape[0], 1)
            z_shp = None

        x_pts = x.points.reshape(x_shp)
        y_pts = y.points.reshape(y_shp)

        x_ones = np.ones(x_shp)
        y_ones = np.ones(y_shp)

        if z_shp:
            z_pts = z.points.reshape(z_shp)
            z_ones = np.ones(z_shp)
        else:
            z_pts = None
            z_ones = None

        return (x_pts, x_ones, y_pts, y_ones, z_pts, z_ones)

    def test_contrived_differential1(self):
        # testing :
        # F = ( cos(lat) cos(lon) )
        # dF/dLon = - sin(lon) cos(lat)     (and to simplify /cos(lat) )
        cube = build_cube(np.empty((30, 60)), spherical=True)

        x = cube.coord("longitude")
        y = cube.coord("latitude")
        y_dim = cube.coord_dims(y)[0]

        cos_x_pts = np.cos(np.radians(x.points)).reshape(1, x.shape[0])
        cos_y_pts = np.cos(np.radians(y.points)).reshape(y.shape[0], 1)

        cube.data = cos_y_pts * cos_x_pts

        lon_coord = x.copy()
        lon_coord.convert_units("radians")
        lat_coord = y.copy()
        lat_coord.convert_units("radians")
        cos_lat_coord = iris.coords.AuxCoord.from_coord(lat_coord)
        cos_lat_coord.points = np.cos(lat_coord.points)
        cos_lat_coord.units = "1"
        cos_lat_coord.rename("cos({})".format(lat_coord.name()))

        temp = iris.analysis.calculus.differentiate(cube, lon_coord)
        df_dlon = iris.analysis.maths.divide(temp, cos_lat_coord, y_dim)

        x = df_dlon.coord("longitude")
        y = df_dlon.coord("latitude")

        sin_x_pts = np.sin(np.radians(x.points)).reshape(1, x.shape[0])
        y_ones = np.ones((y.shape[0], 1))

        data = -sin_x_pts * y_ones
        result = df_dlon.copy(data=data)

        np.testing.assert_array_almost_equal(
            result.data, df_dlon.data, decimal=3
        )

    def test_contrived_differential2(self):
        # testing :
        # w = y^2
        # dw_dy = 2*y
        cube = build_cube(np.empty((10, 30, 60)), spherical=False)

        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(cube)

        w = cube.copy(data=z_ones * x_ones * pow(y_pts, 2.0))

        r = iris.analysis.calculus.differentiate(w, "projection_y_coordinate")

        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(r)
        result = r.copy(data=y_pts * 2.0 * x_ones * z_ones)

        np.testing.assert_array_almost_equal(result.data, r.data, decimal=6)

    def test_contrived_non_spherical_curl1(self):
        # testing :
        # F(x, y, z) = (y, 0, 0)
        # curl( F(x, y, z) ) = (0, 0, -1)

        cube = build_cube(np.empty((25, 50)), spherical=False)

        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(cube)

        u = cube.copy(data=x_ones * y_pts)
        u.rename("u_wind")
        v = cube.copy(data=u.data * 0)
        v.rename("v_wind")

        r = iris.analysis.calculus.curl(u, v)

        # Curl returns None when there is no components of Curl
        self.assertEqual(r[0], None)
        self.assertEqual(r[1], None)
        cube = r[2]
        self.assertCML(
            cube,
            ("analysis", "calculus", "grad_contrived_non_spherical1.cml"),
            checksum=False,
        )
        self.assertTrue(np.all(np.abs(cube.data - (-1.0)) < 1.0e-7))

    def test_contrived_non_spherical_curl2(self):
        # testing :
        # F(x, y, z) = (z^3, x+2, y^2)
        # curl( F(x, y, z) ) = (2y, 3z^2, 1)

        cube = build_cube(np.empty((10, 25, 50)), spherical=False)

        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(cube)

        u = cube.copy(data=pow(z_pts, 3) * x_ones * y_ones)
        v = cube.copy(data=z_ones * (x_pts + 2.0) * y_ones)
        w = cube.copy(data=z_ones * x_ones * pow(y_pts, 2.0))
        u.rename("u_wind")
        v.rename("v_wind")
        w.rename("w_wind")

        r = iris.analysis.calculus.curl(u, v, w)

        # TODO #235 When regridding is not nearest neighbour: the commented out code could be made to work
        # r[0].data should now be tending towards result.data as the resolution of the grid gets higher.
        #        result = r[0].copy(data=True)
        #        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(result)
        #        result.data = y_pts * 2. * x_ones * z_ones
        #        print(repr(r[0].data[0:1, 0:5, 0:25:5]))
        #        print(repr(result.data[0:1, 0:5, 0:25:5]))
        #        np.testing.assert_array_almost_equal(result.data, r[0].data, decimal=2)
        #
        #        result = r[1].copy(data=True)
        #        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(result)
        #        result.data = pow(z_pts, 2) * x_ones * y_ones
        #        np.testing.assert_array_almost_equal(result.data, r[1].data, decimal=6)

        result = r[2].copy()
        result.data = result.data * 0 + 1
        np.testing.assert_array_almost_equal(result.data, r[2].data, decimal=4)

        self.assertCML(
            r,
            ("analysis", "calculus", "curl_contrived_cartesian2.cml"),
            checksum=False,
        )

    def test_contrived_spherical_curl1(self):
        # testing:
        # F(lon, lat, r) = (- r sin(lon), -r cos(lon) sin(lat), 0)
        # curl( F(x, y, z) ) = (0, 0, 0)
        cube = build_cube(np.empty((30, 60)), spherical=True)
        radius = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS

        x = cube.coord("longitude")
        y = cube.coord("latitude")

        cos_x_pts = np.cos(np.radians(x.points)).reshape(1, x.shape[0])
        sin_x_pts = np.sin(np.radians(x.points)).reshape(1, x.shape[0])
        sin_y_pts = np.sin(np.radians(y.points)).reshape(y.shape[0], 1)
        y_ones = np.ones((cube.shape[0], 1))

        u = cube.copy(data=-sin_x_pts * y_ones * radius)
        v = cube.copy(data=-cos_x_pts * sin_y_pts * radius)
        u.rename("u_wind")
        v.rename("v_wind")

        r = iris.analysis.calculus.curl(u, v)[2]

        result = r.copy(data=r.data * 0)

        # Note: This numerical comparison was created when the radius was 1000 times smaller
        np.testing.assert_array_almost_equal(
            result.data[5:-5], r.data[5:-5] / 1000.0, decimal=1
        )
        self.assertCML(
            r, ("analysis", "calculus", "grad_contrived1.cml"), checksum=False
        )

    def test_contrived_spherical_curl2(self):
        # testing:
        # F(lon, lat, r) = (r sin(lat) cos(lon), -r sin(lon), 0)
        # curl( F(x, y, z) ) = (0, 0, -2 cos(lon) cos(lat) )
        cube = build_cube(np.empty((70, 150)), spherical=True)
        radius = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS

        x = cube.coord("longitude")
        y = cube.coord("latitude")

        cos_x_pts = np.cos(np.radians(x.points)).reshape(1, x.shape[0])
        sin_x_pts = np.sin(np.radians(x.points)).reshape(1, x.shape[0])
        cos_y_pts = np.cos(np.radians(y.points)).reshape(y.shape[0], 1)
        sin_y_pts = np.sin(np.radians(y.points)).reshape(y.shape[0], 1)
        y_ones = np.ones((cube.shape[0], 1))

        u = cube.copy(data=sin_y_pts * cos_x_pts * radius)
        v = cube.copy(data=-sin_x_pts * y_ones * radius)
        u.rename("u_wind")
        v.rename("v_wind")

        lon_coord = x.copy()
        lon_coord.convert_units("radians")
        lat_coord = y.copy()
        lat_coord.convert_units("radians")
        cos_lat_coord = iris.coords.AuxCoord.from_coord(lat_coord)
        cos_lat_coord.points = np.cos(lat_coord.points)
        cos_lat_coord.units = "1"
        cos_lat_coord.rename("cos({})".format(lat_coord.name()))

        r = iris.analysis.calculus.curl(u, v)[2]

        x = r.coord("longitude")
        y = r.coord("latitude")

        cos_x_pts = np.cos(np.radians(x.points)).reshape(1, x.shape[0])
        cos_y_pts = np.cos(np.radians(y.points)).reshape(y.shape[0], 1)

        # Expected r-component value: -2 cos(lon) cos(lat)
        result = r.copy(data=-2 * cos_x_pts * cos_y_pts)

        # Note: This numerical comparison was created when the radius was 1000 times smaller
        np.testing.assert_array_almost_equal(
            result.data[30:-30, :], r.data[30:-30, :] / 1000.0, decimal=1
        )
        self.assertCML(
            r, ("analysis", "calculus", "grad_contrived2.cml"), checksum=False
        )


class TestCurlInterface(tests.IrisTest):
    def test_non_conformed(self):
        u = build_cube(np.empty((50, 20)), spherical=True)

        v = u.copy()
        y = v.coord("latitude")
        y.points = y.points + 5
        self.assertRaises(ValueError, iris.analysis.calculus.curl, u, v)

    def test_standard_name(self):
        nx = 20
        ny = 50
        u = build_cube(np.empty((ny, nx)), spherical=True)
        v = u.copy()
        w = u.copy()
        u.rename("u_wind")
        v.rename("v_wind")
        w.rename("w_wind")

        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v)
        self.assertEqual(r, (("u", "v", "w"), "wind"))

        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w)
        self.assertEqual(r, (("u", "v", "w"), "wind"))

        self.assertRaises(
            ValueError,
            iris.analysis.calculus.spatial_vectors_with_phenom_name,
            u,
            None,
            w,
        )
        self.assertRaises(
            ValueError,
            iris.analysis.calculus.spatial_vectors_with_phenom_name,
            None,
            None,
            w,
        )
        self.assertRaises(
            ValueError,
            iris.analysis.calculus.spatial_vectors_with_phenom_name,
            None,
            None,
            None,
        )

        u.rename("x foobar wibble")
        v.rename("y foobar wibble")
        w.rename("z foobar wibble")
        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v)
        self.assertEqual(r, (("x", "y", "z"), "foobar wibble"))

        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w)
        self.assertEqual(r, (("x", "y", "z"), "foobar wibble"))

        u.rename("wibble foobar")
        v.rename("wobble foobar")
        w.rename("tipple foobar")
        #        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w) #should raise a Value Error...
        self.assertRaises(
            ValueError,
            iris.analysis.calculus.spatial_vectors_with_phenom_name,
            u,
            v,
        )
        self.assertRaises(
            ValueError,
            iris.analysis.calculus.spatial_vectors_with_phenom_name,
            u,
            v,
            w,
        )

        u.rename("eastward_foobar")
        v.rename("northward_foobar")
        w.rename("upward_foobar")
        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v)
        self.assertEqual(r, (("eastward", "northward", "upward"), "foobar"))

        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w)
        self.assertEqual(r, (("eastward", "northward", "upward"), "foobar"))

        # Change it to have an inconsistent phenomenon
        v.rename("northward_foobar2")
        self.assertRaises(
            ValueError,
            iris.analysis.calculus.spatial_vectors_with_phenom_name,
            u,
            v,
        )

    def test_rotated_pole(self):
        u = build_cube(np.empty((30, 20)), spherical="rotated")
        v = u.copy()
        u.rename("u_wind")
        v.rename("v_wind")

        x, y, z = iris.analysis.calculus.curl(u, v)
        self.assertEqual(z.coord_system(), u.coord_system())


if __name__ == "__main__":
    unittest.main()
