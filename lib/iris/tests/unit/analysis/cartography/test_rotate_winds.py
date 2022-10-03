# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the function
:func:`iris.analysis.cartography.rotate_winds`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cartopy.crs as ccrs
import numpy as np
import numpy.ma as ma
import pytest

from iris.analysis.cartography import rotate_winds, unrotate_pole
import iris.coord_systems
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


def uv_cubes(x=None, y=None):
    """Return u, v cubes with a grid in a rotated pole CRS."""
    cs = iris.coord_systems.RotatedGeogCS(
        grid_north_pole_latitude=37.5, grid_north_pole_longitude=177.5
    )
    if x is None:
        x = np.linspace(311.9, 391.1, 6)
    if y is None:
        y = np.linspace(-23.6, 24.8, 5)

    x2d, y2d = np.meshgrid(x, y)
    u = 10 * (2 * np.cos(2 * np.deg2rad(x2d) + 3 * np.deg2rad(y2d + 30)) ** 2)
    v = 20 * np.cos(6 * np.deg2rad(x2d))
    lon = DimCoord(
        x, standard_name="grid_longitude", units="degrees", coord_system=cs
    )
    lat = DimCoord(
        y, standard_name="grid_latitude", units="degrees", coord_system=cs
    )
    u_cube = Cube(u, standard_name="x_wind", units="m/s")
    v_cube = Cube(v, standard_name="y_wind", units="m/s")
    for cube in (u_cube, v_cube):
        cube.add_dim_coord(lat.copy(), 0)
        cube.add_dim_coord(lon.copy(), 1)
    return u_cube, v_cube


def uv_cubes_3d(ref_cube, n_realization=3):
    """
    Return 3d u, v cubes with a grid in a rotated pole CRS taken from
    the provided 2d cube, by adding a realization dimension
    coordinate bound to teh zeroth dimension.

    """
    lat = ref_cube.coord("grid_latitude")
    lon = ref_cube.coord("grid_longitude")
    x2d, y2d = np.meshgrid(lon.points, lat.points)
    u = 10 * (2 * np.cos(2 * np.deg2rad(x2d) + 3 * np.deg2rad(y2d + 30)) ** 2)
    v = 20 * np.cos(6 * np.deg2rad(x2d))
    # Multiply slices by factor to give variation over 0th dim.
    factor = np.arange(1, n_realization + 1).reshape(n_realization, 1, 1)
    u = factor * u
    v = factor * v
    realization = DimCoord(np.arange(n_realization), "realization")
    u_cube = Cube(u, standard_name="x_wind", units="m/s")
    v_cube = Cube(v, standard_name="y_wind", units="m/s")
    for cube in (u_cube, v_cube):
        cube.add_dim_coord(realization.copy(), 0)
        cube.add_dim_coord(lat.copy(), 1)
        cube.add_dim_coord(lon.copy(), 2)
    return u_cube, v_cube


class TestPrerequisites(tests.IrisTest):
    def test_different_coord_systems(self):
        u, v = uv_cubes()
        v.coord("grid_latitude").coord_system = iris.coord_systems.GeogCS(1)
        with self.assertRaisesRegex(
            ValueError, "Coordinates differ between u and v cubes"
        ):
            rotate_winds(u, v, iris.coord_systems.OSGB())

    def test_different_xy_coord_systems(self):
        u, v = uv_cubes()
        u.coord("grid_latitude").coord_system = iris.coord_systems.GeogCS(1)
        v.coord("grid_latitude").coord_system = iris.coord_systems.GeogCS(1)
        with self.assertRaisesRegex(
            ValueError, "Coordinate systems of x and y coordinates differ"
        ):
            rotate_winds(u, v, iris.coord_systems.OSGB())

    def test_different_shape(self):
        x = np.linspace(311.9, 391.1, 6)
        y = np.linspace(-23.6, 24.8, 5)
        u, _ = uv_cubes(x, y)
        _, v = uv_cubes(x[:-1], y)
        with self.assertRaisesRegex(ValueError, "same shape"):
            rotate_winds(u, v, iris.coord_systems.OSGB())

    def test_xy_dimensionality(self):
        u, v = uv_cubes()
        # Replace 1d lat with 2d lat.
        x = u.coord("grid_longitude").points
        y = u.coord("grid_latitude").points
        x2d, y2d = np.meshgrid(x, y)
        lat_2d = AuxCoord(
            y2d,
            "grid_latitude",
            units="degrees",
            coord_system=u.coord("grid_latitude").coord_system,
        )
        for cube in (u, v):
            cube.remove_coord("grid_latitude")
            cube.add_aux_coord(lat_2d.copy(), (0, 1))

        with self.assertRaisesRegex(
            ValueError,
            "x and y coordinates must have the same number of dimensions",
        ):
            rotate_winds(u, v, iris.coord_systems.OSGB())

    def test_dim_mapping(self):
        x = np.linspace(311.9, 391.1, 3)
        y = np.linspace(-23.6, 24.8, 3)
        u, v = uv_cubes(x, y)
        v.transpose()
        with self.assertRaisesRegex(ValueError, "Dimension mapping"):
            rotate_winds(u, v, iris.coord_systems.OSGB())


class TestAnalyticComparison(tests.IrisTest):
    @staticmethod
    def _unrotate_equation(
        rotated_lons, rotated_lats, rotated_us, rotated_vs, pole_lon, pole_lat
    ):
        # Perform a rotated-pole 'unrotate winds' transformation on arrays of
        # rotated-lat, rotated-lon, u and v.
        # This can be defined as an analytic function : cf. UMDP015

        # Work out the rotation angles.
        lambda_angle = np.radians(pole_lon - 180.0)
        phi_angle = np.radians(90.0 - pole_lat)

        # Get the locations in true lats+lons.
        trueLongitude, trueLatitude = unrotate_pole(
            rotated_lons, rotated_lats, pole_lon, pole_lat
        )

        # Calculate inter-coordinate rotation coefficients.
        cos_rot = np.cos(np.radians(rotated_lons)) * np.cos(
            np.radians(trueLongitude) - lambda_angle
        ) + np.sin(np.radians(rotated_lons)) * np.sin(
            np.radians(trueLongitude) - lambda_angle
        ) * np.cos(
            phi_angle
        )
        sin_rot = -(
            (
                np.sin(np.radians(trueLongitude) - lambda_angle)
                * np.sin(phi_angle)
            )
            / np.cos(np.radians(rotated_lats))
        )

        # Matrix-multiply to rotate the vectors.
        u_true = rotated_us * cos_rot - rotated_vs * sin_rot
        v_true = rotated_vs * cos_rot + rotated_us * sin_rot

        return u_true, v_true

    def _check_rotated_to_true(self, u_rot, v_rot, target_cs, **kwds):
        # Run test calculation (numeric).
        u_true, v_true = rotate_winds(u_rot, v_rot, target_cs)

        # Perform same calculation via the reference method (equations).
        cs_rot = u_rot.coord("grid_longitude").coord_system
        pole_lat = cs_rot.grid_north_pole_latitude
        pole_lon = cs_rot.grid_north_pole_longitude
        rotated_lons = u_rot.coord("grid_longitude").points
        rotated_lats = u_rot.coord("grid_latitude").points
        rotated_lons_2d, rotated_lats_2d = np.meshgrid(
            rotated_lons, rotated_lats
        )
        rotated_u, rotated_v = u_rot.data, v_rot.data
        u_ref, v_ref = self._unrotate_equation(
            rotated_lons_2d,
            rotated_lats_2d,
            rotated_u,
            rotated_v,
            pole_lon,
            pole_lat,
        )

        # Check that all the numerical results are within given tolerances.
        self.assertArrayAllClose(u_true.data, u_ref, **kwds)
        self.assertArrayAllClose(v_true.data, v_ref, **kwds)

    def test_rotated_to_true__small(self):
        # Check for a small field with varying data.
        target_cs = iris.coord_systems.GeogCS(6371229)
        u_rot, v_rot = uv_cubes()
        self._check_rotated_to_true(
            u_rot, v_rot, target_cs, rtol=1e-5, atol=0.0005
        )

    def test_rotated_to_true_global(self):
        # Check for global fields with various constant wind values
        # - constant in the rotated pole system, that is.
        # We expect less accuracy where this gets close to the true poles.
        target_cs = iris.coord_systems.GeogCS(6371229)
        u_rot, v_rot = uv_cubes(
            x=np.arange(0, 360.0, 15), y=np.arange(-89, 89, 10)
        )
        for vector in ((1, 0), (0, 1), (1, 1), (-3, -1.5)):
            u_rot.data[...] = vector[0]
            v_rot.data[...] = vector[1]
            self._check_rotated_to_true(
                u_rot,
                v_rot,
                target_cs,
                rtol=5e-4,
                atol=5e-4,
                err_msg="vector={}".format(vector),
            )


class TestRotatedToOSGB(tests.IrisTest):
    # Define some coordinate ranges for the uv_cubes 'standard' RotatedPole
    # system, that exceed the OSGB margins, but not by "too much".
    _rp_x_min, _rp_x_max = -5.0, 5.0
    _rp_y_min, _rp_y_max = -5.0, 15.0

    def _uv_cubes_limited_extent(self):
        # Make test cubes suitable for transforming to OSGB, as the standard
        # 'uv_cubes' result goes too far outside, leading to errors.
        x = np.linspace(self._rp_x_min, self._rp_x_max, 6)
        y = np.linspace(self._rp_y_min, self._rp_y_max, 5)
        return uv_cubes(x=x, y=y)

    def test_name(self):
        u, v = self._uv_cubes_limited_extent()
        u.rename("bob")
        v.rename("alice")
        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())
        self.assertEqual(ut.name(), "transformed_" + u.name())
        self.assertEqual(vt.name(), "transformed_" + v.name())

    def test_new_coords(self):
        u, v = self._uv_cubes_limited_extent()
        x = u.coord("grid_longitude").points
        y = u.coord("grid_latitude").points
        x2d, y2d = np.meshgrid(x, y)
        src_crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        tgt_crs = ccrs.OSGB()
        xyz_tran = tgt_crs.transform_points(src_crs, x2d, y2d)

        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())

        points = xyz_tran[..., 0].reshape(x2d.shape)
        expected_x = AuxCoord(
            points,
            standard_name="projection_x_coordinate",
            units="m",
            coord_system=iris.coord_systems.OSGB(),
        )
        self.assertEqual(ut.coord("projection_x_coordinate"), expected_x)
        self.assertEqual(vt.coord("projection_x_coordinate"), expected_x)

        points = xyz_tran[..., 1].reshape(y2d.shape)
        expected_y = AuxCoord(
            points,
            standard_name="projection_y_coordinate",
            units="m",
            coord_system=iris.coord_systems.OSGB(),
        )
        self.assertEqual(ut.coord("projection_y_coordinate"), expected_y)
        self.assertEqual(vt.coord("projection_y_coordinate"), expected_y)

    def test_new_coords_transposed(self):
        u, v = self._uv_cubes_limited_extent()
        # Transpose cubes so that cube is in xy order rather than the
        # typical yx order of meshgrid.
        u.transpose()
        v.transpose()
        x = u.coord("grid_longitude").points
        y = u.coord("grid_latitude").points
        x2d, y2d = np.meshgrid(x, y)
        src_crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        tgt_crs = ccrs.OSGB()
        xyz_tran = tgt_crs.transform_points(src_crs, x2d, y2d)

        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())

        points = xyz_tran[..., 0].reshape(x2d.shape)
        expected_x = AuxCoord(
            points,
            standard_name="projection_x_coordinate",
            units="m",
            coord_system=iris.coord_systems.OSGB(),
        )
        self.assertEqual(ut.coord("projection_x_coordinate"), expected_x)
        self.assertEqual(vt.coord("projection_x_coordinate"), expected_x)

        points = xyz_tran[..., 1].reshape(y2d.shape)
        expected_y = AuxCoord(
            points,
            standard_name="projection_y_coordinate",
            units="m",
            coord_system=iris.coord_systems.OSGB(),
        )
        self.assertEqual(ut.coord("projection_y_coordinate"), expected_y)
        self.assertEqual(vt.coord("projection_y_coordinate"), expected_y)
        # Check dim mapping for 2d coords is yx.
        expected_dims = u.coord_dims("grid_latitude") + u.coord_dims(
            "grid_longitude"
        )
        self.assertEqual(
            ut.coord_dims("projection_x_coordinate"), expected_dims
        )
        self.assertEqual(
            ut.coord_dims("projection_y_coordinate"), expected_dims
        )
        self.assertEqual(
            vt.coord_dims("projection_x_coordinate"), expected_dims
        )
        self.assertEqual(
            vt.coord_dims("projection_y_coordinate"), expected_dims
        )

    def test_orig_coords(self):
        u, v = self._uv_cubes_limited_extent()
        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())
        self.assertEqual(u.coord("grid_latitude"), ut.coord("grid_latitude"))
        self.assertEqual(v.coord("grid_latitude"), vt.coord("grid_latitude"))
        self.assertEqual(u.coord("grid_longitude"), ut.coord("grid_longitude"))
        self.assertEqual(v.coord("grid_longitude"), vt.coord("grid_longitude"))

    def test_magnitude_preservation(self):
        u, v = self._uv_cubes_limited_extent()
        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())
        orig_sq_mag = u.data**2 + v.data**2
        res_sq_mag = ut.data**2 + vt.data**2
        self.assertArrayAllClose(orig_sq_mag, res_sq_mag, rtol=5e-4)

    def test_data_values(self):
        u, v = self._uv_cubes_limited_extent()
        # Slice out 4 points that lie in and outside OSGB extent.
        u = u[1:3, 3:5]
        v = v[1:3, 3:5]
        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())
        # Values precalculated and checked.
        expected_ut_data = np.array(
            [[0.16285514, 0.35323639], [1.82650698, 2.62455840]]
        )
        expected_vt_data = np.array(
            [[19.88979966, 19.01921346], [19.88018847, 19.01424281]]
        )
        # Compare u and v data values against previously calculated values.
        self.assertArrayAllClose(ut.data, expected_ut_data, rtol=1e-5)
        self.assertArrayAllClose(vt.data, expected_vt_data, rtol=1e-5)

    def test_nd_data(self):
        u2d, y2d = self._uv_cubes_limited_extent()
        u, v = uv_cubes_3d(u2d)
        u = u[:, 1:3, 3:5]
        v = v[:, 1:3, 3:5]
        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())
        # Values precalculated and checked (as test_data_values above),
        # then scaled by factor [1, 2, 3] along 0th dim (see uv_cubes_3d()).
        expected_ut_data = np.array(
            [[0.16285514, 0.35323639], [1.82650698, 2.62455840]]
        )
        expected_vt_data = np.array(
            [[19.88979966, 19.01921346], [19.88018847, 19.01424281]]
        )
        factor = np.array([1, 2, 3]).reshape(3, 1, 1)
        expected_ut_data = factor * expected_ut_data
        expected_vt_data = factor * expected_vt_data
        # Compare u and v data values against previously calculated values.
        self.assertArrayAlmostEqual(ut.data, expected_ut_data)
        self.assertArrayAlmostEqual(vt.data, expected_vt_data)

    def test_transposed(self):
        # Test case where the coordinates are not ordered yx in the cube.
        u, v = self._uv_cubes_limited_extent()
        # Slice out 4 points that lie in and outside OSGB extent.
        u = u[1:3, 3:5]
        v = v[1:3, 3:5]
        # Transpose cubes (in-place)
        u.transpose()
        v.transpose()
        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())
        # Values precalculated and checked.
        expected_ut_data = np.array(
            [[0.16285514, 0.35323639], [1.82650698, 2.62455840]]
        ).T
        expected_vt_data = np.array(
            [[19.88979966, 19.01921346], [19.88018847, 19.01424281]]
        ).T
        # Compare u and v data values against previously calculated values.
        self.assertArrayAllClose(ut.data, expected_ut_data, rtol=1e-5)
        self.assertArrayAllClose(vt.data, expected_vt_data, rtol=1e-5)


class TestMasking(tests.IrisTest):
    def test_rotated_to_osgb(self):
        # Rotated Pole data with large extent.
        # A 'correct' answer is not known for this test; it is therefore
        #  written as a 'benchmark' style test - a change in behaviour will
        #  cause a test failure, requiring developers to approve/reject the
        #  new behaviour.
        x = np.linspace(221.9, 301.1, 10)
        y = np.linspace(-23.6, 24.8, 8)
        u, v = uv_cubes(x, y)
        ut, vt = rotate_winds(u, v, iris.coord_systems.OSGB())

        # Ensure cells with discrepancies in magnitude are masked.
        self.assertTrue(ma.isMaskedArray(ut.data))
        self.assertTrue(ma.isMaskedArray(vt.data))

        # Snapshot of mask with fixed tolerance of atol=2e-3
        expected_mask = np.array(
            [
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            ],
            np.bool_,
        )
        self.assertArrayEqual(expected_mask, ut.data.mask)
        self.assertArrayEqual(expected_mask, vt.data.mask)

        # Check unmasked values have sufficiently small error in mag.
        expected_mag = np.sqrt(u.data**2 + v.data**2)
        # Use underlying data to ignore mask in calculation.
        res_mag = np.sqrt(ut.data.data**2 + vt.data.data**2)
        # Calculate percentage error (note there are no zero magnitudes
        # so we can divide safely).
        anom = 100.0 * np.abs(res_mag - expected_mag) / expected_mag
        assert anom[~ut.data.mask].max() == pytest.approx(0.3227935)

    def test_rotated_to_unrotated(self):
        # Suffiently accurate so that no mask is introduced.
        u, v = uv_cubes()
        ut, vt = rotate_winds(u, v, iris.coord_systems.GeogCS(6371229))
        self.assertFalse(ma.isMaskedArray(ut.data))
        self.assertFalse(ma.isMaskedArray(vt.data))


class TestRoundTrip(tests.IrisTest):
    def test_rotated_to_unrotated(self):
        # Check ability to use 2d coords as input.
        u, v = uv_cubes()
        ut, vt = rotate_winds(u, v, iris.coord_systems.GeogCS(6371229))
        # Remove  grid lat and lon, leaving 2d projection coords.
        ut.remove_coord("grid_latitude")
        vt.remove_coord("grid_latitude")
        ut.remove_coord("grid_longitude")
        vt.remove_coord("grid_longitude")
        # Change back.
        orig_cs = u.coord("grid_latitude").coord_system
        res_u, res_v = rotate_winds(ut, vt, orig_cs)
        # Check data values - limited accuracy due to numerical approx.
        self.assertArrayAlmostEqual(res_u.data, u.data, decimal=3)
        self.assertArrayAlmostEqual(res_v.data, v.data, decimal=3)
        # Check coords locations.
        x2d, y2d = np.meshgrid(
            u.coord("grid_longitude").points, u.coord("grid_latitude").points
        )
        # Shift longitude from 0 to 360 -> -180 to 180.
        x2d = np.where(x2d > 180, x2d - 360, x2d)
        res_x = res_u.coord(
            "projection_x_coordinate", coord_system=orig_cs
        ).points
        res_y = res_u.coord(
            "projection_y_coordinate", coord_system=orig_cs
        ).points
        self.assertArrayAlmostEqual(res_x, x2d)
        self.assertArrayAlmostEqual(res_y, y2d)
        res_x = res_v.coord(
            "projection_x_coordinate", coord_system=orig_cs
        ).points
        res_y = res_v.coord(
            "projection_y_coordinate", coord_system=orig_cs
        ).points
        self.assertArrayAlmostEqual(res_x, x2d)
        self.assertArrayAlmostEqual(res_y, y2d)


class TestNonEarthPlanet(tests.IrisTest):
    def test_non_earth_semimajor_axis(self):
        u, v = uv_cubes()
        u.coord("grid_latitude").coord_system = iris.coord_systems.GeogCS(123)
        u.coord("grid_longitude").coord_system = iris.coord_systems.GeogCS(123)
        v.coord("grid_latitude").coord_system = iris.coord_systems.GeogCS(123)
        v.coord("grid_longitude").coord_system = iris.coord_systems.GeogCS(123)
        other_cs = iris.coord_systems.RotatedGeogCS(
            0, 0, ellipsoid=iris.coord_systems.GeogCS(123)
        )
        rotate_winds(u, v, other_cs)


if __name__ == "__main__":
    tests.main()
