# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.analysis.stats.pearsonr` function."""

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import dask
import dask.array
import numpy as np
import numpy.ma as ma
import pytest

import iris
import iris._lazy_data
import iris.analysis.stats as stats
from iris.exceptions import CoordinateNotFoundError


class Mixin:
    def setup_method(self):
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


@tests.skip_data
class TestLazy(Mixin):
    @pytest.fixture
    def mocked_compute(self, monkeypatch):
        m_compute = mock.Mock(wraps=dask.base.compute)

        # The three dask compute functions are all the same function but monkeypatch
        # does not automatically know that.
        # https://stackoverflow.com/questions/77820437
        monkeypatch.setattr(dask.base, dask.base.compute.__name__, m_compute)
        monkeypatch.setattr(dask, dask.compute.__name__, m_compute)
        monkeypatch.setattr(dask.array, dask.array.compute.__name__, m_compute)

        return m_compute

    def test_perfect_corr(self, mocked_compute):
        r = stats.pearsonr(self.cube_a, self.cube_a, ["latitude", "longitude"])
        mocked_compute.assert_not_called()
        np.testing.assert_array_equal(r.data, np.array([1.0] * 6))

    def test_perfect_corr_all_dims(self, mocked_compute):
        r = stats.pearsonr(self.cube_a, self.cube_a)
        mocked_compute.assert_not_called()
        np.testing.assert_array_equal(r.data, np.array([1.0]))

    def test_compatible_cubes(self, mocked_compute):
        r = stats.pearsonr(self.cube_a, self.cube_b, ["latitude", "longitude"])
        mocked_compute.assert_not_called()
        np.testing.assert_array_almost_equal(
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

    def test_broadcast_cubes(self, mocked_compute):
        r1 = stats.pearsonr(
            self.cube_a, self.cube_b[0, :, :], ["latitude", "longitude"]
        )
        r2 = stats.pearsonr(
            self.cube_b[0, :, :], self.cube_a, ["latitude", "longitude"]
        )

        mocked_compute.assert_not_called()
        r_by_slice = [
            stats.pearsonr(
                self.cube_a[i, :, :],
                self.cube_b[0, :, :],
                ["latitude", "longitude"],
            ).data
            for i in range(6)
        ]
        np.testing.assert_array_equal(r1.data, np.array(r_by_slice))
        np.testing.assert_array_equal(r2.data, np.array(r_by_slice))

    def test_compatible_cubes_weighted(self, mocked_compute):
        r = stats.pearsonr(
            self.cube_a, self.cube_b, ["latitude", "longitude"], self.weights
        )

        mocked_compute.assert_not_called()
        np.testing.assert_array_almost_equal(
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

    def test_broadcast_cubes_weighted(self, mocked_compute):
        r = stats.pearsonr(
            self.cube_a,
            self.cube_b[0, :, :],
            ["latitude", "longitude"],
            weights=self.weights[0, :, :],
        )

        mocked_compute.assert_not_called()
        r_by_slice = [
            stats.pearsonr(
                self.cube_a[i, :, :],
                self.cube_b[0, :, :],
                ["latitude", "longitude"],
                weights=self.weights[0, :, :],
            ).data
            for i in range(6)
        ]
        np.testing.assert_array_almost_equal(r.data, np.array(r_by_slice))

    def test_broadcast_transpose_cubes_weighted(self, mocked_compute):
        # Reference is calculated with no transposition.
        r_ref = stats.pearsonr(
            self.cube_a,
            self.cube_b[0, :, :],
            ["latitude", "longitude"],
            weights=self.weights[0, :, :],
        )

        self.cube_a.transpose()
        r_test = stats.pearsonr(
            self.cube_a,
            self.cube_b[0, :, :],
            ["latitude", "longitude"],
            weights=self.weights[0, :, :],
        )

        mocked_compute.assert_not_called()
        # Should get the same result, but transposed.
        np.testing.assert_array_almost_equal(r_test.data, r_ref.data.T)

    def test_weight_error(self):
        with pytest.raises(ValueError):
            stats.pearsonr(
                self.cube_a,
                self.cube_b[0, :, :],
                ["latitude", "longitude"],
                weights=self.weights,
            )

    def test_mdtol(self, mocked_compute):
        cube_small = self.cube_a[:, 0, 0]
        cube_small_masked = iris.util.mask_cube(cube_small, [0, 0, 0, 1, 1, 1])
        r1 = stats.pearsonr(cube_small, cube_small_masked)
        r2 = stats.pearsonr(cube_small, cube_small_masked, mdtol=0.49)

        mocked_compute.assert_not_called()
        np.testing.assert_array_almost_equal(r1.data, np.array([0.74586593]))
        tests.assert_masked_array_equal(r2.data, ma.array([0], mask=[True]))

    def test_common_mask_simple(self, mocked_compute):
        cube_small = self.cube_a[:, 0, 0]
        cube_small_masked = iris.util.mask_cube(cube_small, [0, 0, 0, 1, 1, 1])
        r = stats.pearsonr(cube_small, cube_small_masked, common_mask=True)

        mocked_compute.assert_not_called()
        np.testing.assert_array_almost_equal(r.data, np.array([1.0]))

    def test_common_mask_broadcast(self, mocked_compute):
        cube_small = iris.util.mask_cube(self.cube_a[:, 0, 0], [0, 0, 0, 0, 0, 1])
        mask_2d = np.zeros((6, 2), dtype=bool)
        # 2d mask varies on unshared coord:
        mask_2d[0, 1] = 1

        # Make a (6, 2) cube.
        cube_small_2d = self.cube_a[:, 0:2, 0]
        # Duplicate data along unshared coord's dimension.
        new_data = iris.util.broadcast_to_shape(
            cube_small.core_data(), (6, 2), dim_map=[0]
        )
        cube_small_2d.data = iris.util._mask_array(new_data, mask_2d)

        r = stats.pearsonr(
            cube_small,
            cube_small_2d,
            weights=self.weights[:, 0, 0],
            common_mask=True,
        )

        mocked_compute.assert_not_called()
        np.testing.assert_array_almost_equal(r.data, np.array([1.0, 1.0]))
        # 2d mask does not vary on unshared coord:
        cube_small_2d.data.mask[0, 0] = 1
        r = stats.pearsonr(cube_small, cube_small_2d, common_mask=True)
        np.testing.assert_array_almost_equal(r.data, np.array([1.0, 1.0]))


class TestReal(TestLazy):
    def setup_method(self):
        super().setup_method()
        for cube in [self.cube_a, self.cube_b]:
            _ = cube.data


class TestCoordHandling(Mixin):
    def test_lenient_handling(self):
        # Smoke test that mismatched var_name does not prevent operation.
        self.cube_a.coord("time").var_name = "wibble"
        stats.pearsonr(self.cube_a, self.cube_b)

    def test_incompatible_cubes(self):
        with pytest.raises(ValueError):
            stats.pearsonr(self.cube_a[:, 0, :], self.cube_b[0, :, :], "longitude")

    def test_single_coord(self):
        # Smoke test that single coord can be passed as single string.
        stats.pearsonr(self.cube_a, self.cube_b, "latitude")

    def test_non_existent_coord(self):
        with pytest.raises(CoordinateNotFoundError):
            stats.pearsonr(self.cube_a, self.cube_b, "bad_coord")
