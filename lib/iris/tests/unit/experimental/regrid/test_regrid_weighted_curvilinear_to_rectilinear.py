# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function
:func:`iris.experimental.regrid.regrid_weighted_curvilinear_to_rectilinear`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import copy

import numpy as np
import numpy.ma as ma

import iris
from iris.coord_systems import GeogCS, LambertConformal
import iris.coords
from iris.coords import AuxCoord, DimCoord
import iris.cube
from iris.experimental.regrid import (
    regrid_weighted_curvilinear_to_rectilinear as regrid,
)
from iris.fileformats.pp import EARTH_RADIUS

PLAIN_LATLON_CS = GeogCS(EARTH_RADIUS)


class Test(tests.IrisTest):
    def setUp(self):
        # Source cube.
        self.test_src_name = "air_temperature"
        self.test_src_units = "K"
        self.test_src_data = ma.arange(1, 13, dtype=np.float64).reshape(3, 4)
        self.test_src_attributes = dict(wibble="wobble")
        self.test_scalar_coord = iris.coords.DimCoord(
            [1], long_name="test_scalar_coord"
        )
        self.src = iris.cube.Cube(
            self.test_src_data,
            standard_name=self.test_src_name,
            units=self.test_src_units,
            aux_coords_and_dims=[(self.test_scalar_coord, None)],
            attributes=self.test_src_attributes,
        )

        # Source cube x-coordinates.
        points = np.array(
            [[10, 20, 200, 220], [110, 120, 180, 185], [190, 203, 211, 220]]
        )
        self.src_x_positive = iris.coords.AuxCoord(
            points,
            standard_name="longitude",
            units="degrees",
            coord_system=PLAIN_LATLON_CS,
        )
        self.src_x_transpose = iris.coords.AuxCoord(
            points.T,
            standard_name="longitude",
            units="degrees",
            coord_system=PLAIN_LATLON_CS,
        )
        points = np.array(
            [
                [-180, -176, -170, -150],
                [-180, -179, -178, -177],
                [-170, -168, -159, -140],
            ]
        )
        self.src_x_negative = iris.coords.AuxCoord(
            points,
            standard_name="longitude",
            units="degrees",
            coord_system=PLAIN_LATLON_CS,
        )

        # Source cube y-coordinates.
        points = np.array([[0, 4, 3, 1], [5, 7, 10, 6], [12, 20, 15, 30]])
        self.src_y = iris.coords.AuxCoord(
            points,
            standard_name="latitude",
            units="degrees",
            coord_system=PLAIN_LATLON_CS,
        )
        self.src_y_transpose = iris.coords.AuxCoord(
            points.T,
            standard_name="latitude",
            units="degrees",
            coord_system=PLAIN_LATLON_CS,
        )

        # Weights.
        self.weight_factor = 10
        self.weights = np.asarray(self.test_src_data) * self.weight_factor

        # Target grid cube.
        self.grid = iris.cube.Cube(np.zeros((2, 2)))

        # Target grid cube x-coordinates.
        self.grid_x_inc = iris.coords.DimCoord(
            [187, 200],
            standard_name="longitude",
            units="degrees",
            bounds=[[180, 190], [190, 220]],
            coord_system=PLAIN_LATLON_CS,
        )
        self.grid_x_dec = iris.coords.DimCoord(
            [200, 187],
            standard_name="longitude",
            units="degrees",
            bounds=[[220, 190], [190, 180]],
            coord_system=PLAIN_LATLON_CS,
        )

        # Target grid cube y-coordinates.
        self.grid_y_inc = iris.coords.DimCoord(
            [2, 10],
            standard_name="latitude",
            units="degrees",
            bounds=[[0, 5], [5, 30]],
            coord_system=PLAIN_LATLON_CS,
        )
        self.grid_y_dec = iris.coords.DimCoord(
            [10, 2],
            standard_name="latitude",
            units="degrees",
            bounds=[[30, 5], [5, 0]],
            coord_system=PLAIN_LATLON_CS,
        )

    def _weighted_mean(self, points):
        points = np.asarray(points, dtype=np.float64)
        weights = points * self.weight_factor
        numerator = denominator = 0
        for point, weight in zip(points, weights):
            numerator += point * weight
            denominator += weight
        return numerator / denominator

    def _expected_cube(self, data):
        cube = iris.cube.Cube(data)
        cube.metadata = copy.deepcopy(self.src.metadata)
        grid_x = self.grid.coord(axis="x")
        grid_y = self.grid.coord(axis="y")
        cube.add_dim_coord(grid_x.copy(), self.grid.coord_dims(grid_x))
        cube.add_dim_coord(grid_y.copy(), self.grid.coord_dims(grid_y))
        src_x = self.src.coord(axis="x")
        src_y = self.src.coord(axis="y")
        for coord in self.src.aux_coords:
            if coord is not src_x and coord is not src_y:
                if not self.src.coord_dims(coord):
                    cube.add_aux_coord(coord)
        return cube

    def test_aligned_src_x(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array(
            [
                0,
                self._weighted_mean([3]),
                self._weighted_mean([7, 8]),
                self._weighted_mean([9, 10, 11]),
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[True, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_non_latlon(self):
        odd_coord_system = LambertConformal()
        co_src_y = AuxCoord(
            self.src_y.points,
            standard_name="projection_y_coordinate",
            units="km",
            coord_system=odd_coord_system,
        )
        co_src_x = AuxCoord(
            self.src_x_positive.points,
            standard_name="projection_x_coordinate",
            units="km",
            coord_system=odd_coord_system,
        )
        co_grid_y = DimCoord(
            self.grid_y_inc.points,
            bounds=self.grid_y_inc.bounds,
            standard_name="projection_y_coordinate",
            units="km",
            coord_system=odd_coord_system,
        )
        co_grid_x = DimCoord(
            self.grid_x_inc.points,
            bounds=self.grid_x_inc.bounds,
            standard_name="projection_x_coordinate",
            units="km",
            coord_system=odd_coord_system,
        )
        self.src.add_aux_coord(co_src_y, (0, 1))
        self.src.add_aux_coord(co_src_x, (0, 1))
        self.grid.add_dim_coord(co_grid_y, 0)
        self.grid.add_dim_coord(co_grid_x, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array(
            [
                0,
                self._weighted_mean([3]),
                self._weighted_mean([7, 8]),
                self._weighted_mean([9, 10, 11]),
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[True, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_src_xy_not_2d(self):
        new_shape = (2, 2, 3)
        # Reshape the source cube, including the X and Y coordinates,
        # from (3, 4) to (2, 2, 3).
        # This is really an "invalid" reshape, but should still work because
        # the XY shape is actually irrelevant to the regrid operation.
        src = iris.cube.Cube(
            self.test_src_data.reshape(new_shape),
            standard_name=self.test_src_name,
            units=self.test_src_units,
            aux_coords_and_dims=[(self.test_scalar_coord, None)],
            attributes=self.test_src_attributes,
        )
        co_src_y = self.src_y.copy(points=self.src_y.points.reshape(new_shape))
        co_src_x = self.src_x_positive.copy(
            points=self.src_x_positive.points.reshape(new_shape)
        )
        src.add_aux_coord(co_src_y, (0, 1, 2))
        src.add_aux_coord(co_src_x, (0, 1, 2))
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        weights = self.weights.reshape(new_shape)
        result = regrid(src, weights, self.grid)
        # NOTE: set the grid of self.src to make '_expected_cube' work ...
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        # ... given that, we expect exactly the same 'normal' result.
        data = np.array(
            [
                0,
                self._weighted_mean([3]),
                self._weighted_mean([7, 8]),
                self._weighted_mean([9, 10, 11]),
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[True, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_aligned_src_x_mask(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.src.data[([1, 2, 2], [3, 0, 2])] = ma.masked
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array(
            [
                0,
                self._weighted_mean([3]),
                self._weighted_mean([7]),
                self._weighted_mean([10]),
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[True, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_aligned_src_x_zero_weights(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        self.weights[:, 2] = 0
        self.weights[1, :] = 0
        result = regrid(self.src, self.weights, self.grid)
        data = np.array([0, 0, 0, self._weighted_mean([9, 10])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[True, True], [True, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_aligned_tgt_dec(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.grid.add_dim_coord(self.grid_y_dec, 0)
        self.grid.add_dim_coord(self.grid_x_dec, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array(
            [
                self._weighted_mean([10, 11, 12]),
                self._weighted_mean([8, 9]),
                self._weighted_mean([3, 4]),
                0,
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[False, False], [False, True]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_misaligned_src_x_negative(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_negative, (0, 1))
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array(
            [
                self._weighted_mean([1, 2]),
                self._weighted_mean([3, 4]),
                self._weighted_mean([5, 6, 7, 8]),
                self._weighted_mean([9, 10, 11]),
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[False, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_misaligned_src_x_negative_mask(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_negative, (0, 1))
        self.src.data[([0, 0, 1, 1, 2, 2], [1, 3, 1, 3, 1, 3])] = ma.masked
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array(
            [
                self._weighted_mean([1]),
                self._weighted_mean([3]),
                self._weighted_mean([5, 7]),
                self._weighted_mean([9, 11]),
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[False, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_misaligned_tgt_dec(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_negative, (0, 1))
        self.grid.add_dim_coord(self.grid_y_dec, 0)
        self.grid.add_dim_coord(self.grid_x_dec, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array(
            [
                self._weighted_mean([10, 11, 12]),
                self._weighted_mean([6, 7, 8, 9]),
                self._weighted_mean([4]),
                self._weighted_mean([2, 3]),
            ]
        ).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[False, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)


if __name__ == "__main__":
    tests.main()
