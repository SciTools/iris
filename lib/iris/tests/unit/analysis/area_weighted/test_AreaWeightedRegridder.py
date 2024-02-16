# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.analysis._area_weighted.AreaWeightedRegridder`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
from unittest import mock

import numpy as np

from iris import load_cube
from iris.analysis._area_weighted import (
    AreaWeightedRegridder,
    _regrid_area_weighted_rectilinear_src_and_grid__prepare,
)
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube


class Test(tests.IrisTest):
    def cube(self, x, y):
        data = np.arange(len(x) * len(y)).reshape(len(y), len(x))
        cube = Cube(data)
        lat = DimCoord(y, "latitude", units="degrees")
        lon = DimCoord(x, "longitude", units="degrees")
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
        return cube

    def grids(self):
        src = self.cube(np.linspace(20, 30, 3), np.linspace(10, 25, 4))
        target = self.cube(np.linspace(22, 28, 8), np.linspace(11, 22, 9))
        return src, target

    def extract_grid(self, cube):
        return cube.coord("latitude"), cube.coord("longitude")

    def check_mdtol(self, mdtol=None):
        src_grid, target_grid = self.grids()
        # Get _regrid_info result
        _regrid_info = _regrid_area_weighted_rectilinear_src_and_grid__prepare(
            src_grid, target_grid
        )
        self.assertEqual(len(_regrid_info), 9)
        with mock.patch(
            "iris.analysis._area_weighted."
            "_regrid_area_weighted_rectilinear_src_and_grid__prepare",
            return_value=_regrid_info,
        ) as prepare:
            with mock.patch(
                "iris.analysis._area_weighted."
                "_regrid_area_weighted_rectilinear_src_and_grid__perform",
                return_value=mock.sentinel.result,
            ) as perform:
                # Setup the regridder
                if mdtol is None:
                    regridder = AreaWeightedRegridder(src_grid, target_grid)
                    mdtol = 1
                else:
                    regridder = AreaWeightedRegridder(
                        src_grid, target_grid, mdtol=mdtol
                    )
                # Now regrid the source cube
                src = src_grid
                result = regridder(src)

                # Make a new cube to regrid with different data so we can
                # distinguish between regridding the original src grid
                # definition cube and the cube passed to the regridder.
                src = src_grid.copy()
                src.data += 10
                result = regridder(src)

        # Prepare:
        self.assertEqual(prepare.call_count, 1)
        _, args, kwargs = prepare.mock_calls[0]
        self.assertEqual(self.extract_grid(args[1]), self.extract_grid(target_grid))

        # Perform:
        self.assertEqual(perform.call_count, 2)
        _, args, kwargs = perform.mock_calls[1]
        self.assertEqual(args[0], src)
        self.assertEqual(kwargs, {"mdtol": mdtol})
        self.assertIs(result, mock.sentinel.result)

    def test_default(self):
        self.check_mdtol()

    def test_specified_mdtol(self):
        self.check_mdtol(0.5)

    def test_invalid_high_mdtol(self):
        src, target = self.grids()
        msg = "mdtol must be in range 0 - 1"
        with self.assertRaisesRegex(ValueError, msg):
            AreaWeightedRegridder(src, target, mdtol=1.2)

    def test_invalid_low_mdtol(self):
        src, target = self.grids()
        msg = "mdtol must be in range 0 - 1"
        with self.assertRaisesRegex(ValueError, msg):
            AreaWeightedRegridder(src, target, mdtol=-0.2)

    def test_mismatched_src_coord_systems(self):
        src = Cube(np.zeros((3, 4)))
        cs = GeogCS(6543210)
        lat = DimCoord(np.arange(3), "latitude", coord_system=cs)
        lon = DimCoord(np.arange(4), "longitude")
        src.add_dim_coord(lat, 0)
        src.add_dim_coord(lon, 1)
        target = mock.Mock()
        with self.assertRaises(ValueError):
            AreaWeightedRegridder(src, target)

    def test_src_and_target_are_the_same(self):
        src = self.cube(np.linspace(20, 30, 3), np.linspace(10, 25, 4))
        target = self.cube(np.linspace(20, 30, 3), np.linspace(10, 25, 4))
        regridder = AreaWeightedRegridder(src, target)
        result = regridder(src)
        self.assertArrayAllClose(result.data, target.data)

    def test_multiple_src_on_same_grid(self):
        coord_names = ["latitude", "longitude"]
        src1 = self.cube(np.linspace(20, 32, 4), np.linspace(10, 22, 4))
        src2 = self.cube(np.linspace(20, 32, 4), np.linspace(10, 22, 4))
        src2.data *= 4
        self.assertArrayEqual(src1.data * 4, src2.data)
        for name in coord_names:
            # Remove coords system and units so it is no longer spherical.
            src1.coord(name).coord_system = None
            src1.coord(name).units = None
            src2.coord(name).coord_system = None
            src2.coord(name).units = None

        target = self.cube(np.linspace(20, 32, 2), np.linspace(10, 22, 2))
        # Ensure the bounds of the target cover the same range as the
        # source.
        target_lat_bounds = np.column_stack(
            (
                src1.coord("latitude").bounds[[0, 1], [0, 1]],
                src1.coord("latitude").bounds[[2, 3], [0, 1]],
            )
        )
        target.coord("latitude").bounds = target_lat_bounds
        target_lon_bounds = np.column_stack(
            (
                src1.coord("longitude").bounds[[0, 1], [0, 1]],
                src1.coord("longitude").bounds[[2, 3], [0, 1]],
            )
        )
        target.coord("longitude").bounds = target_lon_bounds
        for name in coord_names:
            # Remove coords system and units so it is no longer spherical.
            target.coord(name).coord_system = None
            target.coord(name).units = None

        regridder = AreaWeightedRegridder(src1, target)
        result1 = regridder(src1)
        result2 = regridder(src2)

        reference1 = self.cube(np.linspace(20, 32, 2), np.linspace(10, 22, 2))
        reference1.data = np.array(
            [
                [np.mean(src1.data[0:2, 0:2]), np.mean(src1.data[0:2, 2:4])],
                [np.mean(src1.data[2:4, 0:2]), np.mean(src1.data[2:4, 2:4])],
            ]
        )
        reference1.coord("latitude").bounds = target_lat_bounds
        reference1.coord("longitude").bounds = target_lon_bounds

        reference2 = self.cube(np.linspace(20, 32, 2), np.linspace(10, 22, 2))
        reference2.data = np.array(
            [
                [np.mean(src2.data[0:2, 0:2]), np.mean(src2.data[0:2, 2:4])],
                [np.mean(src2.data[2:4, 0:2]), np.mean(src2.data[2:4, 2:4])],
            ]
        )
        reference2.coord("latitude").bounds = target_lat_bounds
        reference2.coord("longitude").bounds = target_lon_bounds

        for name in coord_names:
            # Remove coords system and units so it is no longer spherical.
            reference1.coord(name).coord_system = None
            reference1.coord(name).units = None
            reference2.coord(name).coord_system = None
            reference2.coord(name).units = None

        # Compare the cubes rather than just the data.
        self.assertEqual(result1, reference1)
        self.assertEqual(result2, reference2)

    def test_src_data_different_dims(self):
        src, target = self.grids()
        regridder = AreaWeightedRegridder(src, target)
        result = regridder(src)
        expected_mean, expected_std = 4.772097735195653, 2.211698479817678
        self.assertArrayShapeStats(result, (9, 8), expected_mean, expected_std)
        # New source cube with additional "levels" dimension
        # Each level has identical x-y data so the mean and std stats remain
        # identical when x, y and z dims are reordered
        levels = DimCoord(np.arange(5), "model_level_number")
        lat = src.coord("latitude")
        lon = src.coord("longitude")
        data = np.repeat(src.data[np.newaxis, ...], 5, axis=0)
        src = Cube(data)
        src.add_dim_coord(levels, 0)
        src.add_dim_coord(lat, 1)
        src.add_dim_coord(lon, 2)
        result = regridder(src)
        self.assertArrayShapeStats(result, (5, 9, 8), expected_mean, expected_std)
        # Check data with dims in different order
        # Reshape src so that the coords are ordered [x, z, y],
        # the mean and std statistics should be the same
        data = np.moveaxis(src.data.copy(), 2, 0)
        src = Cube(data)
        src.add_dim_coord(lon, 0)
        src.add_dim_coord(levels, 1)
        src.add_dim_coord(lat, 2)
        result = regridder(src)
        self.assertArrayShapeStats(result, (8, 5, 9), expected_mean, expected_std)
        # Check data with dims in different order
        # Reshape src so that the coords are ordered [y, x, z],
        # the mean and std statistics should be the same
        data = np.moveaxis(src.data.copy(), 2, 0)
        src = Cube(data)
        src.add_dim_coord(lat, 0)
        src.add_dim_coord(lon, 1)
        src.add_dim_coord(levels, 2)
        result = regridder(src)
        self.assertArrayShapeStats(result, (9, 8, 5), expected_mean, expected_std)


@tests.skip_data
class TestLazy(tests.IrisTest):
    # Setup
    def setUp(self) -> None:
        # Prepare a cube and a template
        cube_file_path = tests.get_data_path(["NetCDF", "regrid", "regrid_xyt.nc"])
        self.cube = load_cube(cube_file_path)

        template_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_template_global_latlon.nc"]
        )
        self.template_cube = load_cube(template_file_path)

        # Chunked data makes the regridder run repeatedly
        self.cube.data = self.cube.lazy_data().rechunk((1, -1, -1))

    def test_src_stays_lazy(self) -> None:
        cube = self.cube.copy()
        # Regrid the cube onto the template.
        regridder = AreaWeightedRegridder(cube, self.template_cube)
        regridder(cube)
        # Base cube stays lazy
        self.assertTrue(cube.has_lazy_data())

    def test_output_lazy(self) -> None:
        cube = self.cube.copy()
        # Regrid the cube onto the template.
        regridder = AreaWeightedRegridder(cube, self.template_cube)
        out = regridder(cube)
        # Lazy base cube means lazy output
        self.assertTrue(out.has_lazy_data())


if __name__ == "__main__":
    tests.main()
