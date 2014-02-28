# (C) British Crown Copyright 2013 - 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Test function
:func:`iris.experimental.regrid.regrid_weighted_curvilinear_to_rectilinear`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import copy

import numpy as np
import numpy.ma as ma

import iris
import iris.coords
import iris.cube
from iris.experimental.regrid \
    import regrid_weighted_curvilinear_to_rectilinear as regrid


class Test(tests.IrisTest):
    def setUp(self):
        # Source cube.
        data = ma.arange(1, 13, dtype=np.float).reshape(3, 4)
        attributes = dict(wibble='wobble')
        bibble = iris.coords.DimCoord([1], long_name='bibble')
        self.src = iris.cube.Cube(data,
                                  standard_name='air_temperature',
                                  units='K',
                                  aux_coords_and_dims=[(bibble, None)],
                                  attributes=attributes)

        # Source cube x-coordinates.
        points = np.array([[010, 020, 200, 220],
                           [110, 120, 180, 185],
                           [190, 203, 211, 220]])
        self.src_x_positive = iris.coords.AuxCoord(points,
                                                   standard_name='longitude',
                                                   units='degrees')
        self.src_x_transpose = iris.coords.AuxCoord(points.T,
                                                    standard_name='longitude',
                                                    units='degrees')
        points = np.array([[-180, -176, -170, -150],
                           [-180, -179, -178, -177],
                           [-170, -168, -159, -140]])
        self.src_x_negative = iris.coords.AuxCoord(points,
                                                   standard_name='longitude',
                                                   units='degrees')

        # Source cube y-coordinates.
        points = np.array([[00, 04, 03, 01],
                           [05, 07, 10, 06],
                           [12, 20, 15, 30]])
        self.src_y = iris.coords.AuxCoord(points,
                                          standard_name='latitude',
                                          units='degrees')
        self.src_y_transpose = iris.coords.AuxCoord(points.T,
                                                    standard_name='latitude',
                                                    units='degrees')

        # Weights.
        self.weight_factor = 10
        self.weights = np.asarray(data) * self.weight_factor

        # Target grid cube.
        self.grid = iris.cube.Cube(np.zeros((2, 2)))

        # Target grid cube x-coordinates.
        self.grid_x_inc = iris.coords.DimCoord([187, 200],
                                               standard_name='longitude',
                                               units='degrees',
                                               bounds=[[180, 190],
                                                       [190, 220]])
        self.grid_x_dec = iris.coords.DimCoord([200, 187],
                                               standard_name='longitude',
                                               units='degrees',
                                               bounds=[[220, 190],
                                                       [190, 180]])

        # Target grid cube y-coordinates.
        self.grid_y_inc = iris.coords.DimCoord([2, 10],
                                               standard_name='latitude',
                                               units='degrees',
                                               bounds=[[0, 5],
                                                       [5, 30]])
        self.grid_y_dec = iris.coords.DimCoord([10, 2],
                                               standard_name='latitude',
                                               units='degrees',
                                               bounds=[[30, 5],
                                                       [5, 0]])

    def _weighted_mean(self, points):
        points = np.asarray(points, dtype=np.float)
        weights = points * self.weight_factor
        numerator = denominator = 0
        for point, weight in zip(points, weights):
            numerator += point * weight
            denominator += weight
        return numerator / denominator

    def _expected_cube(self, data):
        cube = iris.cube.Cube(data)
        cube.metadata = copy.deepcopy(self.src)
        grid_x = self.grid.coord('longitude')
        grid_y = self.grid.coord('latitude')
        cube.add_dim_coord(grid_x.copy(), self.grid.coord_dims(grid_x))
        cube.add_dim_coord(grid_y.copy(), self.grid.coord_dims(grid_y))
        src_x = self.src.coord('longitude')
        src_y = self.src.coord('latitude')
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
        data = np.array([0,
                         self._weighted_mean([3]),
                         self._weighted_mean([7, 8]),
                         self._weighted_mean([9, 10, 11])]).reshape(2, 2)
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
        data = np.array([0,
                         self._weighted_mean([3]),
                         self._weighted_mean([7]),
                         self._weighted_mean([10])]).reshape(2, 2)
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

    def test_aligned_src_x_transpose(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_transpose, (1, 0))
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array([0,
                         self._weighted_mean([3]),
                         self._weighted_mean([7, 8]),
                         self._weighted_mean([9, 10, 11])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[True, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_aligned_src_y_transpose(self):
        self.src.add_aux_coord(self.src_y_transpose, (1, 0))
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array([0,
                         self._weighted_mean([3]),
                         self._weighted_mean([7, 8]),
                         self._weighted_mean([9, 10, 11])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[True, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_aligned_tgt_dec(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.grid.add_dim_coord(self.grid_y_dec, 0)
        self.grid.add_dim_coord(self.grid_x_dec, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array([self._weighted_mean([10, 11, 12]),
                         self._weighted_mean([8, 9]),
                         self._weighted_mean([3, 4]),
                         0]).reshape(2, 2)
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
        data = np.array([self._weighted_mean([1, 2]),
                         self._weighted_mean([3, 4]),
                         self._weighted_mean([5, 6, 7, 8]),
                         self._weighted_mean([9, 10, 11])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[False, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)

    def test_misaligned_src_x_negative_mask(self):
        self.src.add_aux_coord(self.src_y, (0, 1))
        self.src.add_aux_coord(self.src_x_negative, (0, 1))
        self.src.data[([0, 0, 1, 1, 2, 2],
                       [1, 3, 1, 3, 1, 3])] = ma.masked
        self.grid.add_dim_coord(self.grid_y_inc, 0)
        self.grid.add_dim_coord(self.grid_x_inc, 1)
        result = regrid(self.src, self.weights, self.grid)
        data = np.array([self._weighted_mean([1]),
                         self._weighted_mean([3]),
                         self._weighted_mean([5, 7]),
                         self._weighted_mean([9, 11])]).reshape(2, 2)
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
        data = np.array([self._weighted_mean([10, 11, 12]),
                         self._weighted_mean([6, 7, 8, 9]),
                         self._weighted_mean([4]),
                         self._weighted_mean([2, 3])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)
        mask = np.array([[False, False], [False, False]])
        self.assertArrayEqual(result.data.mask, mask)


if __name__ == '__main__':
    tests.main()
