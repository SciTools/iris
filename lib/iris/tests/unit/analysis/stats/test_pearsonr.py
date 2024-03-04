# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.analysis.stats.pearsonr` function."""

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

import iris
import iris.analysis.stats as stats
from iris.exceptions import CoordinateNotFoundError


@tests.skip_data
class Test(tests.IrisTest):
    def setUp(self):
        # 3D cubes:
        cube_temp = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
            )
        )
        self.cube_a = cube_temp[0:6]
        self.cube_b = cube_temp[20:26]
        self.cube_b.replace_coord(self.cube_a.coord("time").copy())
        cube_temp = self.cube_a.copy()
        cube_temp.coord("latitude").guess_bounds()
        cube_temp.coord("longitude").guess_bounds()
        self.weights = iris.analysis.cartography.area_weights(cube_temp)

    def test_perfect_corr(self):
        r = stats.pearsonr(self.cube_a, self.cube_a, ["latitude", "longitude"])
        self.assertArrayEqual(r.data, np.array([1.0] * 6))

    def test_perfect_corr_all_dims(self):
        r = stats.pearsonr(self.cube_a, self.cube_a)
        self.assertArrayEqual(r.data, np.array([1.0]))

    def test_incompatible_cubes(self):
        with self.assertRaises(ValueError):
            stats.pearsonr(
                self.cube_a[:, 0, :], self.cube_b[0, :, :], "longitude"
            )

    def test_compatible_cubes(self):
        r = stats.pearsonr(self.cube_a, self.cube_b, ["latitude", "longitude"])
        self.assertArrayAlmostEqual(
            r.data,
            [
                0.81114936,
                0.81690538,
                0.79833135,
                0.81118674,
                0.79745386,
                0.81278484,
            ],
        )

    def test_broadcast_cubes(self):
        r1 = stats.pearsonr(
            self.cube_a, self.cube_b[0, :, :], ["latitude", "longitude"]
        )
        r2 = stats.pearsonr(
            self.cube_b[0, :, :], self.cube_a, ["latitude", "longitude"]
        )
        r_by_slice = [
            stats.pearsonr(
                self.cube_a[i, :, :],
                self.cube_b[0, :, :],
                ["latitude", "longitude"],
            ).data
            for i in range(6)
        ]
        self.assertArrayEqual(r1.data, np.array(r_by_slice))
        self.assertArrayEqual(r2.data, np.array(r_by_slice))

    def test_compatible_cubes_weighted(self):
        r = stats.pearsonr(
            self.cube_a, self.cube_b, ["latitude", "longitude"], self.weights
        )
        self.assertArrayAlmostEqual(
            r.data,
            [
                0.79105429,
                0.79988078,
                0.78825089,
                0.79925653,
                0.79009810,
                0.80115292,
            ],
        )

    def test_broadcast_cubes_weighted(self):
        r = stats.pearsonr(
            self.cube_a,
            self.cube_b[0, :, :],
            ["latitude", "longitude"],
            weights=self.weights[0, :, :],
        )
        r_by_slice = [
            stats.pearsonr(
                self.cube_a[i, :, :],
                self.cube_b[0, :, :],
                ["latitude", "longitude"],
                weights=self.weights[0, :, :],
            ).data
            for i in range(6)
        ]
        self.assertArrayAlmostEqual(r.data, np.array(r_by_slice))

    def test_weight_error(self):
        with self.assertRaises(ValueError):
            stats.pearsonr(
                self.cube_a,
                self.cube_b[0, :, :],
                ["latitude", "longitude"],
                weights=self.weights,
            )

    def test_non_existent_coord(self):
        with self.assertRaises(CoordinateNotFoundError):
            stats.pearsonr(self.cube_a, self.cube_b, "bad_coord")

    def test_mdtol(self):
        cube_small = self.cube_a[:, 0, 0]
        cube_small_masked = cube_small.copy()
        cube_small_masked.data = ma.array(
            cube_small.data, mask=np.array([0, 0, 0, 1, 1, 1], dtype=bool)
        )
        r1 = stats.pearsonr(cube_small, cube_small_masked)
        r2 = stats.pearsonr(cube_small, cube_small_masked, mdtol=0.49)
        self.assertArrayAlmostEqual(r1.data, np.array([0.74586593]))
        self.assertMaskedArrayEqual(r2.data, ma.array([0], mask=[True]))

    def test_common_mask_simple(self):
        cube_small = self.cube_a[:, 0, 0]
        cube_small_masked = cube_small.copy()
        cube_small_masked.data = ma.array(
            cube_small.data, mask=np.array([0, 0, 0, 1, 1, 1], dtype=bool)
        )
        r = stats.pearsonr(cube_small, cube_small_masked, common_mask=True)
        self.assertArrayAlmostEqual(r.data, np.array([1.0]))

    def test_common_mask_broadcast(self):
        cube_small = self.cube_a[:, 0, 0]
        cube_small_2d = self.cube_a[:, 0:2, 0]
        cube_small.data = ma.array(
            cube_small.data, mask=np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        )
        cube_small_2d.data = ma.array(
            np.tile(cube_small.data[:, np.newaxis], 2),
            mask=np.zeros((6, 2), dtype=bool),
        )
        # 2d mask varies on unshared coord:
        cube_small_2d.data.mask[0, 1] = 1
        r = stats.pearsonr(
            cube_small,
            cube_small_2d,
            weights=self.weights[:, 0, 0],
            common_mask=True,
        )
        self.assertArrayAlmostEqual(r.data, np.array([1.0, 1.0]))
        # 2d mask does not vary on unshared coord:
        cube_small_2d.data.mask[0, 0] = 1
        r = stats.pearsonr(cube_small, cube_small_2d, common_mask=True)
        self.assertArrayAlmostEqual(r.data, np.array([1.0, 1.0]))


if __name__ == "__main__":
    tests.main()
