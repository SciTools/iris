# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:class:`iris.analysis.trajectory.UnstructuredNearestNeigbourRegridder`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.analysis.trajectory import (
    UnstructuredNearestNeigbourRegridder as unn_gridder,
)
from iris.coord_systems import GeogCS, RotatedGeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList


class MixinExampleSetup:
    # Common code for regridder test classes.

    def setUp(self):
        # Basic test values.
        src_x_y_value = np.array(
            [
                [20.12, 11.73, 0.01],
                [120.23, -20.73, 1.12],
                [290.34, 33.88, 2.23],
                [-310.45, 57.8, 3.34],
            ]
        )
        tgt_grid_x = np.array([-173.2, -100.3, -32.5, 1.4, 46.6, 150.7])
        tgt_grid_y = np.array([-80.1, -30.2, 0.3, 47.4, 75.5])

        # Make sample 1-D source cube.
        src = Cube(src_x_y_value[:, 2])
        src.add_aux_coord(
            AuxCoord(
                src_x_y_value[:, 0], standard_name="longitude", units="degrees"
            ),
            0,
        )
        src.add_aux_coord(
            AuxCoord(
                src_x_y_value[:, 1], standard_name="latitude", units="degrees"
            ),
            0,
        )
        self.src_cube = src

        # Make sample grid cube.
        grid = Cube(np.zeros(tgt_grid_y.shape + tgt_grid_x.shape))
        grid.add_dim_coord(
            DimCoord(tgt_grid_y, standard_name="latitude", units="degrees"), 0
        )
        grid.add_dim_coord(
            DimCoord(tgt_grid_x, standard_name="longitude", units="degrees"), 1
        )
        self.grid_cube = grid

        # Make expected-result, from the expected source-index at each point.
        expected_result_indices = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 2, 0, 0, 0, 1],
                [1, 2, 2, 0, 0, 1],
                [3, 2, 2, 3, 3, 3],
                [3, 2, 3, 3, 3, 3],
            ]
        )
        self.expected_data = self.src_cube.data[expected_result_indices]

        # Make a 3D source cube, based on the existing 2d test data.
        z_cubes = [src.copy() for _ in range(3)]
        for i_z, z_cube in enumerate(z_cubes):
            z_cube.add_aux_coord(DimCoord([i_z], long_name="z"))
            z_cube.data = z_cube.data + 100.0 * i_z
        self.src_z_cube = CubeList(z_cubes).merge_cube()

        # Make a corresponding 3d expected result.
        self.expected_data_zxy = self.src_z_cube.data[
            :, expected_result_indices
        ]

    def _check_expected(
        self,
        src_cube=None,
        grid_cube=None,
        expected_data=None,
        expected_coord_names=None,
    ):
        # Test regridder creation + operation against expected results.
        if src_cube is None:
            src_cube = self.src_cube
        if grid_cube is None:
            grid_cube = self.grid_cube
        gridder = unn_gridder(src_cube, grid_cube)
        result = gridder(src_cube)
        if expected_coord_names is not None:
            # Check result coordinate identities.
            self.assertEqual(
                [coord.name() for coord in result.coords()],
                expected_coord_names,
            )
        if expected_data is None:
            # By default, check against the 'standard' data result.
            expected_data = self.expected_data
        self.assertArrayEqual(result.data, expected_data)
        return result


class Test__init__(MixinExampleSetup, tests.IrisTest):
    # Exercise all the constructor argument checks.

    def test_fail_no_src_x(self):
        self.src_cube.remove_coord("longitude")
        msg_re = "Source cube must have X- and Y-axis coordinates"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_no_src_y(self):
        self.src_cube.remove_coord("latitude")
        msg_re = "Source cube must have X- and Y-axis coordinates"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_bad_src_dims(self):
        self.src_cube = self.grid_cube
        msg_re = "Source.*same cube dimensions"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_mixed_latlons(self):
        self.src_cube.coord("longitude").rename("projection_x_coordinate")
        msg_re = "any.*latitudes/longitudes.*all must be"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_bad_latlon_units(self):
        self.grid_cube.coord("longitude").units = "m"
        msg_re = 'does not convert to "degrees"'
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_non_latlon_units_mismatch(self):
        # Convert all to non-latlon system (does work: see in "Test__call__").
        for cube in (self.src_cube, self.grid_cube):
            for axis_name in ("x", "y"):
                coord = cube.coord(axis=axis_name)
                coord_name = "projection_{}_coordinate".format(axis_name)
                coord.rename(coord_name)
                coord.units = "m"
        # Change one of the output units.
        self.grid_cube.coord(axis="x").units = "1"
        msg_re = "Source and target.*must have the same units"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_no_tgt_x(self):
        self.grid_cube.remove_coord("longitude")
        msg_re = "must contain a single 1D x coordinate"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_no_tgt_y(self):
        self.grid_cube.remove_coord("latitude")
        msg_re = "must contain a single 1D y coordinate"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_src_cs_mismatch(self):
        cs = GeogCS(1000.0)
        self.src_cube.coord("latitude").coord_system = cs
        msg_re = "must all have the same coordinate system"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_tgt_cs_mismatch(self):
        cs = GeogCS(1000.0)
        self.grid_cube.coord("latitude").coord_system = cs
        msg_re = "x.*and y.*must have the same coordinate system"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)

    def test_fail_src_tgt_cs_mismatch(self):
        cs = GeogCS(1000.0)
        self.src_cube.coord("latitude").coord_system = cs
        self.src_cube.coord("longitude").coord_system = cs
        msg_re = "Source and target.*same coordinate system"
        with self.assertRaisesRegex(ValueError, msg_re):
            unn_gridder(self.src_cube, self.grid_cube)


class Test__call__(MixinExampleSetup, tests.IrisTest):
    # Test regridder operation and results.

    def test_basic_latlon(self):
        # Check a test operation.
        self._check_expected(
            expected_coord_names=["latitude", "longitude"],
            expected_data=self.expected_data,
        )

    def test_non_latlon(self):
        # Check different answer in cartesian coordinates (no wrapping, etc).
        # Convert to non-latlon system, with the same coord values.
        for cube in (self.src_cube, self.grid_cube):
            for axis_name in ("x", "y"):
                coord = cube.coord(axis=axis_name)
                coord_name = "projection_{}_coordinate".format(axis_name)
                coord.rename(coord_name)
                coord.units = "m"
        # Check for a somewhat different result.
        non_latlon_indices = np.array(
            [
                [3, 0, 0, 0, 1, 1],
                [3, 0, 0, 0, 0, 1],
                [3, 0, 0, 0, 0, 1],
                [3, 0, 0, 0, 0, 1],
                [3, 0, 0, 0, 0, 1],
            ]
        )
        expected_data = self.src_cube.data[non_latlon_indices]
        self._check_expected(expected_data=expected_data)

    def test_multidimensional_xy(self):
        # Recast the 4-point source cube as 2*2 : should yield the same result.
        co_x = self.src_cube.coord(axis="x")
        co_y = self.src_cube.coord(axis="y")
        new_src = Cube(self.src_cube.data.reshape((2, 2)))
        new_x_co = AuxCoord(
            co_x.points.reshape((2, 2)),
            standard_name="longitude",
            units="degrees",
        )
        new_y_co = AuxCoord(
            co_y.points.reshape((2, 2)),
            standard_name="latitude",
            units="degrees",
        )
        new_src.add_aux_coord(new_x_co, (0, 1))
        new_src.add_aux_coord(new_y_co, (0, 1))
        self._check_expected(src_cube=new_src)

    def test_transposed_grid(self):
        # Show that changing the order of the grid X and Y has no effect.
        new_grid_cube = self.grid_cube.copy()
        new_grid_cube.transpose((1, 0))
        # Check that the new grid is in (X, Y) order.
        self.assertEqual(
            [coord.name() for coord in new_grid_cube.coords()],
            ["longitude", "latitude"],
        )
        # Check that the result is the same, dimension order is still Y,X.
        self._check_expected(
            grid_cube=new_grid_cube,
            expected_coord_names=["latitude", "longitude"],
        )

    def test_compatible_source(self):
        # Check operation on data with different dimensions to the original
        # source cube for the regridder creation.
        gridder = unn_gridder(self.src_cube, self.grid_cube)
        result = gridder(self.src_z_cube)
        self.assertEqual(
            [coord.name() for coord in result.coords()],
            ["z", "latitude", "longitude"],
        )
        self.assertArrayEqual(result.data, self.expected_data_zxy)

    def test_fail_incompatible_source(self):
        # Check that a slightly modified source cube is *not* acceptable.
        modified_src_cube = self.src_cube.copy()
        points = modified_src_cube.coord(axis="x").points
        points[0] += 0.01
        modified_src_cube.coord(axis="x").points = points
        gridder = unn_gridder(self.src_cube, self.grid_cube)
        msg = "not defined on the same source grid"
        with self.assertRaisesRegex(ValueError, msg):
            gridder(modified_src_cube)

    def test_transposed_source(self):
        # Check operation on data where the 'trajectory' dimension is not the
        # last one.
        src_z_cube = self.src_z_cube
        src_z_cube.transpose((1, 0))
        self._check_expected(
            src_cube=src_z_cube, expected_data=self.expected_data_zxy
        )

    def test_radians_degrees(self):
        # Check source + target unit conversions, grid and result in degrees.
        for axis_name in ("x", "y"):
            self.src_cube.coord(axis=axis_name).convert_units("radians")
            self.grid_cube.coord(axis=axis_name).convert_units("degrees")
        result = self._check_expected()
        self.assertEqual(result.coord(axis="x").units, "degrees")

    def test_degrees_radians(self):
        # Check source + target unit conversions, grid and result in radians.
        for axis_name in ("x", "y"):
            self.src_cube.coord(axis=axis_name).convert_units("degrees")
            self.grid_cube.coord(axis=axis_name).convert_units("radians")
        result = self._check_expected()
        self.assertEqual(result.coord(axis="x").units, "radians")

    def test_alternative_cs(self):
        # Check the result is just the same in a different coordinate system.
        cs = RotatedGeogCS(
            grid_north_pole_latitude=75.3,
            grid_north_pole_longitude=102.5,
            ellipsoid=GeogCS(100.0),
        )
        for cube in (self.src_cube, self.grid_cube):
            for coord_name in ("longitude", "latitude"):
                cube.coord(coord_name).coord_system = cs
        self._check_expected()


if __name__ == "__main__":
    tests.main()
