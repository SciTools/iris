# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :class:`iris.analysis._area_weighted.AreaWeightedRegridder`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

import numpy as np

from iris.analysis._area_weighted import AreaWeightedRegridder
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
        return cube

    def grids(self):
        src = self.cube(np.linspace(20, 30, 3), np.linspace(10, 25, 4))
        target = self.cube(np.linspace(6, 18, 8), np.linspace(11, 22, 9))
        return src, target

    def extract_grid(self, cube):
        return cube.coord("latitude"), cube.coord("longitude")

    def check_mdtol(self, mdtol=None):
        src_grid, target_grid = self.grids()
        with mock.patch(
            "iris.experimental.regrid."
            "_regrid_area_weighted_rectilinear_src_and_grid__prepare"
        ) as prepare:
            if mdtol is None:
                regridder = AreaWeightedRegridder(src_grid, target_grid)
                mdtol = 1
            else:
                regridder = AreaWeightedRegridder(
                    src_grid, target_grid, mdtol=mdtol
                )

        # Make a new cube to regrid with different data so we can
        # distinguish between regridding the original src grid
        # definition cube and the cube passed to the regridder.
        src = src_grid.copy()
        src.data += 10

        with mock.patch(
            "iris.experimental.regrid."
            "_regrid_area_weighted_rectilinear_src_and_grid__perform",
            return_value=mock.sentinel.result,
        ) as perform:
            result = regridder(src)

        # Prepare:
        self.assertEqual(prepare.call_count, 1)
        _, args, kwargs = prepare.mock_calls[0]
        self.assertEqual(
            self.extract_grid(args[1]), self.extract_grid(target_grid)
        )

        # Perform:
        self.assertEqual(perform.call_count, 1)
        _, args, kwargs = perform.mock_calls[0]
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
        for name in ["latitude", "longitude"]:
            src.coord(name).guess_bounds()
            target.coord(name).guess_bounds()
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
            # Ensure contiguous bounds exists.
            src1.coord(name).guess_bounds()
            src2.coord(name).guess_bounds()

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

    def test_mismatched_data_dims(self):
        coord_names = ["latitude", "longitude"]
        x = np.linspace(20, 32, 4)
        y = np.linspace(10, 22, 4)
        src1 = self.cube(x, y)

        data = np.arange(len(y) * len(x)).reshape(len(x), len(y))
        src2 = Cube(data)
        lat = DimCoord(y, "latitude", units="degrees")
        lon = DimCoord(x, "longitude", units="degrees")
        # Add dim coords in opposite order to self.cube.
        src2.add_dim_coord(lat, 1)
        src2.add_dim_coord(lon, 0)
        for name in coord_names:
            # Ensure contiguous bounds exists.
            src1.coord(name).guess_bounds()
            src2.coord(name).guess_bounds()

        target = self.cube(np.linspace(20, 32, 2), np.linspace(10, 22, 2))
        for name in coord_names:
            # Ensure the bounds of the target cover the same range as the
            # source.
            target.coord(name).bounds = np.column_stack(
                (
                    src1.coord(name).bounds[[0, 1], [0, 1]],
                    src1.coord(name).bounds[[2, 3], [0, 1]],
                )
            )

        regridder = AreaWeightedRegridder(src1, target)
        self.assertRaises(ValueError, regridder, src2)


if __name__ == "__main__":
    tests.main()
