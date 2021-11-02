# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the function
:func:`iris.analysis.cartography.gridcell_angles`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from cf_units import Unit
import numpy as np

from iris.analysis.cartography import gridcell_angles
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.tests.stock import lat_lon_cube, sample_2d_latlons


def _2d_multicells_testcube(cellsize_degrees=1.0):
    """
    Create a test cube with a grid of X and Y points, where each gridcell
    is independent (disjoint), arranged at an angle == the x-coord point.

    """
    # Setup np.linspace arguments to make the coordinate points.
    x0, x1, nx = -164, 164, 9
    y0, y1, ny = -75, 75, 7

    lats = np.linspace(y0, y1, ny, endpoint=True)
    lons_angles = np.linspace(x0, x1, nx, endpoint=True)
    x_pts_2d, y_pts_2d = np.meshgrid(lons_angles, lats)

    # Make gridcells rectangles surrounding these centrepoints, but also
    # tilted at various angles (= same as x-point lons, as that's easy).

    # Calculate centrepoint lons+lats : in radians, and shape (ny, nx, 1).
    xangs, yangs = np.deg2rad(x_pts_2d), np.deg2rad(y_pts_2d)
    xangs, yangs = [arr[..., None] for arr in (xangs, yangs)]
    # Program which corners are up+down on each gridcell axis.
    dx_corners = [[[-1, 1, 1, -1]]]
    dy_corners = [[[-1, -1, 1, 1]]]
    # Calculate the relative offsets in x+y at the 4 corners.
    x_ofs_2d = cellsize_degrees * np.cos(xangs) * dx_corners
    x_ofs_2d -= cellsize_degrees * np.sin(xangs) * dy_corners
    y_ofs_2d = cellsize_degrees * np.cos(xangs) * dy_corners
    y_ofs_2d += cellsize_degrees * np.sin(xangs) * dx_corners
    # Apply a latitude stretch to make correct angles on the globe.
    y_ofs_2d *= np.cos(yangs)
    # Make bounds arrays by adding the corner offsets to the centrepoints.
    x_bds_2d = x_pts_2d[..., None] + x_ofs_2d
    y_bds_2d = y_pts_2d[..., None] + y_ofs_2d

    # Create a cube with these points + bounds in its 'X' and 'Y' coords.
    co_x = AuxCoord(
        points=x_pts_2d,
        bounds=x_bds_2d,
        standard_name="longitude",
        units="degrees",
    )
    co_y = AuxCoord(
        points=y_pts_2d,
        bounds=y_bds_2d,
        standard_name="latitude",
        units="degrees",
    )
    cube = Cube(np.zeros((ny, nx)))
    cube.add_aux_coord(co_x, (0, 1))
    cube.add_aux_coord(co_y, (0, 1))
    return cube


class TestGridcellAngles(tests.IrisTest):
    def setUp(self):
        # Make a small "normal" contiguous-bounded cube to test on.
        # This one is regional.
        self.standard_regional_cube = sample_2d_latlons(
            regional=True, transformed=True
        )
        # Record the standard correct angle answers.
        result_cube = gridcell_angles(self.standard_regional_cube)
        result_cube.convert_units("degrees")
        self.standard_result_cube = result_cube
        self.standard_small_cube_results = result_cube.data

    def _check_multiple_orientations_and_latitudes(
        self,
        method="mid-lhs, mid-rhs",
        atol_degrees=0.005,
        cellsize_degrees=1.0,
    ):

        cube = _2d_multicells_testcube(cellsize_degrees=cellsize_degrees)

        # Calculate gridcell angles at each point.
        angles_cube = gridcell_angles(cube, cell_angle_boundpoints=method)

        # Check that the results are a close match to the original intended
        # gridcell orientation angles.
        # NOTE: neither the above gridcell construction nor the calculation
        # itself are exact :  Errors scale as the square of gridcell sizes.
        angles_cube.convert_units("degrees")
        angles_calculated = angles_cube.data

        # Note: the gridcell angles **should** just match the longitudes at
        # each point
        angles_expected = cube.coord("longitude").points

        # Wrap both into standard range for comparison.
        angles_calculated = (angles_calculated + 360.0) % 360.0
        angles_expected = (angles_expected + 360.0) % 360.0

        # Assert (toleranced) equality, and return results.
        self.assertArrayAllClose(
            angles_calculated, angles_expected, atol=atol_degrees
        )

        return angles_calculated, angles_expected

    def test_various_orientations_and_locations(self):
        self._check_multiple_orientations_and_latitudes()

    def test_result_form(self):
        # Check properties of the result cube *other than* the data values.
        test_cube = self.standard_regional_cube
        result_cube = self.standard_result_cube
        self.assertEqual(
            result_cube.long_name, "gridcell_angle_from_true_east"
        )
        self.assertEqual(result_cube.units, Unit("degrees"))
        self.assertEqual(len(result_cube.coords()), 2)
        self.assertEqual(
            result_cube.coord(axis="x"), test_cube.coord(axis="x")
        )
        self.assertEqual(
            result_cube.coord(axis="y"), test_cube.coord(axis="y")
        )

    def test_bottom_edge_method(self):
        # Get results with the "other" calculation method + check to tolerance.
        # A smallish cellsize should yield similar results in both cases.
        r1, _ = self._check_multiple_orientations_and_latitudes()
        r2, _ = self._check_multiple_orientations_and_latitudes(
            method="lower-left, lower-right",
            cellsize_degrees=0.1,
            atol_degrees=0.1,
        )

        # Not *exactly* the same : this checks we tested the 'other' method !
        self.assertFalse(np.allclose(r1, r2))
        # Note: results are a bit different in places.  This is acceptable.
        self.assertArrayAllClose(r1, r2, atol=0.1)

    def test_bounded_coord_args(self):
        # Check that passing the coords gives the same result as the cube.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        result = gridcell_angles(co_x, co_y)
        self.assertArrayAllClose(result.data, self.standard_small_cube_results)

    def test_coords_radians_args(self):
        # Check it still works with coords converted to radians.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        for coord in (co_x, co_y):
            coord.convert_units("radians")
        result = gridcell_angles(co_x, co_y)
        self.assertArrayAllClose(result.data, self.standard_small_cube_results)

    def test_bounds_array_args(self):
        # Check we can calculate from bounds values alone.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        # Results drawn from coord bounds should be nearly the same,
        # but not exactly, because of the different 'midpoint' values.
        result = gridcell_angles(co_x.bounds, co_y.bounds)
        self.assertArrayAllClose(
            result.data, self.standard_small_cube_results, atol=0.1
        )

    def test_unbounded_regional_coord_args(self):
        # Remove the coord bounds to check points-based calculation.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        for coord in (co_x, co_y):
            coord.bounds = None
        result = gridcell_angles(co_x, co_y)
        # Note: in this case, we can expect the leftmost and rightmost columns
        # to be rubbish, because the data is not global.
        # But the rest should match okay.
        self.assertArrayAllClose(
            result.data[:, 1:-1], self.standard_small_cube_results[:, 1:-1]
        )

    def test_points_array_args(self):
        # Check we can calculate from points arrays alone (no coords).
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        # As previous, the leftmost and rightmost columns are not good.
        result = gridcell_angles(co_x.points, co_y.points)
        self.assertArrayAllClose(
            result.data[:, 1:-1], self.standard_small_cube_results[:, 1:-1]
        )

    def test_unbounded_global(self):
        # For a contiguous global grid, a result based on points, i.e. with the
        # bounds removed, should be a reasonable match for the 'ideal' one
        # based on the bounds.

        # Make a global cube + calculate ideal bounds-based results.
        global_cube = sample_2d_latlons(transformed=True)
        result_cube = gridcell_angles(global_cube)
        result_cube.convert_units("degrees")
        global_cube_results = result_cube.data

        # Check a points-based calculation on the same basic grid.
        co_x, co_y = (global_cube.coord(axis=ax) for ax in ("x", "y"))
        for coord in (co_x, co_y):
            coord.bounds = None
        result = gridcell_angles(co_x, co_y)
        # In this case, the match is actually rather poor (!).
        self.assertArrayAllClose(result.data, global_cube_results, atol=7.5)
        # Leaving off first + last columns again gives a decent result.
        self.assertArrayAllClose(
            result.data[:, 1:-1], global_cube_results[:, 1:-1]
        )

        # NOTE: although this looks just as bad as 'test_points_array_args',
        # maximum errors there in the end columns are actually > 100 degrees !

    def test_nonlatlon_coord_system(self):
        # Check with points specified in an unexpected coord system.
        cube = sample_2d_latlons(regional=True, rotated=True)
        result = gridcell_angles(cube)
        self.assertArrayAllClose(result.data, self.standard_small_cube_results)
        # Check that the result has transformed (true-latlon) coordinates.
        self.assertEqual(len(result.coords()), 2)
        x_coord = result.coord(axis="x")
        y_coord = result.coord(axis="y")
        self.assertEqual(x_coord.shape, cube.shape)
        self.assertEqual(y_coord.shape, cube.shape)
        self.assertIsNotNone(cube.coord_system)
        self.assertIsNone(x_coord.coord_system)
        self.assertIsNone(y_coord.coord_system)

    def test_fail_coords_bad_units(self):
        # Check error with bad coords units.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        co_y.units = "m"
        with self.assertRaisesRegex(ValueError, "must have angular units"):
            gridcell_angles(co_x, co_y)

    def test_fail_nonarraylike(self):
        # Check error with bad args.
        co_x, co_y = 1, 2
        with self.assertRaisesRegex(
            ValueError, "must have array shape property"
        ):
            gridcell_angles(co_x, co_y)

    def test_fail_non2d_coords(self):
        # Check error with bad args.
        cube = lat_lon_cube()
        with self.assertRaisesRegex(
            ValueError, "inputs must have 2-dimensional shape"
        ):
            gridcell_angles(cube)

    def test_fail_different_shapes(self):
        # Check error with mismatched shapes.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        co_y = co_y[1:]
        with self.assertRaisesRegex(ValueError, "must have same shape"):
            gridcell_angles(co_x, co_y)

    def test_fail_different_coord_system(self):
        # Check error with mismatched coord systems.
        cube = sample_2d_latlons(regional=True, rotated=True)
        cube.coord(axis="x").coord_system = None
        with self.assertRaisesRegex(
            ValueError, "must have same coordinate system"
        ):
            gridcell_angles(cube)

    def test_fail_cube_dims(self):
        # Check error with mismatched cube dims.
        cube = self.standard_regional_cube
        # Make 5x6 into 5x5.
        cube = cube[:, :-1]
        co_x = cube.coord(axis="x")
        pts, bds = co_x.points, co_x.bounds
        co_new_x = co_x.copy(
            points=pts.transpose((1, 0)), bounds=bds.transpose((1, 0, 2))
        )
        cube.remove_coord(co_x)
        cube.add_aux_coord(co_new_x, (1, 0))
        with self.assertRaisesRegex(
            ValueError, "must have the same cube dimensions"
        ):
            gridcell_angles(cube)

    def test_fail_coord_noncoord(self):
        # Check that passing a coord + an array gives an error.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        with self.assertRaisesRegex(
            ValueError, "is a Coordinate, but .* is not"
        ):
            gridcell_angles(co_x, co_y.bounds)

    def test_fail_noncoord_coord(self):
        # Check that passing an array + a coord gives an error.
        co_x, co_y = (
            self.standard_regional_cube.coord(axis=ax) for ax in ("x", "y")
        )
        with self.assertRaisesRegex(
            ValueError, "is a Coordinate, but .* is not"
        ):
            gridcell_angles(co_x.points, co_y)

    def test_fail_bad_method(self):
        with self.assertRaisesRegex(
            ValueError, "unrecognised cell_angle_boundpoints"
        ):
            self._check_multiple_orientations_and_latitudes(
                method="something_unknown"
            )


if __name__ == "__main__":
    tests.main()
