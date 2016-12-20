# (C) British Crown Copyright 2016, Met Office
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
Unit tests for
:class:`iris.analysis.trajectory.UnstructuredNearestNeigbourRegridder`.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
import iris.tests.stock

from iris.analysis.trajectory import \
    UnstructuredNearestNeigbourRegridder as unn_gridder


class Test__init__(tests.IrisTest):
    pass


class Test__call__(tests.IrisTest):
    def setUp(self):
        # Basic test values.
        src_x_y_value = np.array([
            [20.12, 11.73, 0.01],
            [120.23, -20.73, 1.12],
            [290.34, 33.88, 2.23],
            [-310.45, 57.8, 3.34]])
        tgt_grid_x = np.array([-173.2, -100.3, -32.5, 1.4, 46.6, 150.7])
        tgt_grid_y = np.array([-80.1, -30.2, 0.3, 47.4, 75.5])

        # Make sample 1-D source cube.
        src = Cube(src_x_y_value[:, 2])
        src.add_aux_coord(AuxCoord(src_x_y_value[:, 0],
                                   standard_name='longitude', units='degrees'),
                          0)
        src.add_aux_coord(AuxCoord(src_x_y_value[:, 1],
                                   standard_name='latitude', units='degrees'),
                          0)
        self.src_cube = src

        # Make sample grid cube.
        grid = Cube(np.zeros(tgt_grid_y.shape + tgt_grid_x.shape))
        grid.add_dim_coord(DimCoord(tgt_grid_y,
                                    standard_name='latitude', units='degrees'),
                           0)
        grid.add_dim_coord(DimCoord(tgt_grid_x,
                                    standard_name='longitude',
                                    units='degrees'),
                           1)
        self.grid_cube = grid

        # Record expected source-index for each point.
        expected_result_indices = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 1],
            [1, 2, 2, 0, 0, 1],
            [3, 2, 2, 3, 3, 3],
            [3, 2, 3, 3, 3, 3]])
        self.expected_data = self.src_cube.data[expected_result_indices]

        # 3D source data, based on the existing.
        z_cubes = [src.copy() for _ in range(3)]
        for i_z, z_cube in enumerate(z_cubes):
            z_cube.add_aux_coord(DimCoord([i_z], long_name='z'))
            z_cube.data = z_cube.data + 100.0 * i_z
        self.src_z_cube = CubeList(z_cubes).merge_cube()
        self.expected_data_zxy = \
            self.src_z_cube.data[:, expected_result_indices]

    def _check_expected(self, src_cube=None, grid_cube=None,
                        expected_data=None,
                        expected_coord_names=None):
        if src_cube is None:
            src_cube = self.src_cube
        if grid_cube is None:
            grid_cube = self.grid_cube
        gridder = unn_gridder(src_cube, grid_cube)
        result = gridder(src_cube)
        if expected_coord_names is not None:
            # Check result coordinate identities.
            self.assertEqual([coord.name() for coord in result.coords()],
                             expected_coord_names)
        if expected_data is None:
            # By default, check against the 'standard' data result.
            expected_data = self.expected_data
        self.assertArrayEqual(result.data, expected_data)
        return result

    def test_basic_latlon(self):
        # Check a test operation.
        self._check_expected(expected_coord_names=['latitude', 'longitude'],
                             expected_data=self.expected_data)

    def test_non_latlon(self):
        # Check different in cartesian coordinates (no wrapping, etc).
        for cube in (self.src_cube, self.grid_cube):
            cube.coord(axis='x').rename('projection_x_coordinate')
            cube.coord(axis='y').rename('projection_y_coordinate')
        non_latlon_indices = np.array([
            [3, 0, 0, 0, 1, 1],
            [3, 0, 0, 0, 0, 1],
            [3, 0, 0, 0, 0, 1],
            [3, 0, 0, 0, 0, 1],
            [3, 0, 0, 0, 0, 1]])
        expected_data = self.src_cube.data[non_latlon_indices]
        self._check_expected(expected_data=expected_data)

    def test_multidimensional_xy(self):
        # Recast the 4-point source cube as 2*2 : should yield the same result.
        co_x = self.src_cube.coord(axis='x')
        co_y = self.src_cube.coord(axis='y')
        new_src = Cube(self.src_cube.data.reshape((2, 2)))
        new_x_co = AuxCoord(co_x.points.reshape((2, 2)),
                            standard_name='longitude', units='degrees')
        new_y_co = AuxCoord(co_y.points.reshape((2, 2)),
                            standard_name='latitude', units='degrees')
        new_src.add_aux_coord(new_x_co, (0, 1))
        new_src.add_aux_coord(new_y_co, (0, 1))
        self._check_expected(src_cube=new_src)

    def test_transposed_grid(self):
        # Show that changing the order of the grid X and Y has no effect.
        new_grid_cube = self.grid_cube.copy()
        new_grid_cube.transpose((1, 0))
        # Check that the new grid is in (X, Y) order.
        self.assertEqual([coord.name() for coord in new_grid_cube.coords()],
                         ['longitude', 'latitude'])
        # Check that the result is the same, dimension order is still Y,X.
        self._check_expected(grid_cube=new_grid_cube,
                             expected_coord_names=['latitude', 'longitude'])

    def test_compatible_source(self):
        # Check operation on data with different dimensions to the original
        # source cube for the regridder creation.
        gridder = unn_gridder(self.src_cube, self.grid_cube)
        result = gridder(self.src_z_cube)
        self.assertEqual([coord.name() for coord in result.coords()],
                         ['z', 'latitude', 'longitude'])
        self.assertArrayEqual(result.data, self.expected_data_zxy)

    def test_transposed_source(self):
        # Check operation on data where the 'trajectory' dimension is not the
        # last one.
        src_z_cube = self.src_z_cube
        src_z_cube.transpose((1, 0))
        self._check_expected(src_cube=src_z_cube,
                             expected_data=self.expected_data_zxy)

    def test_radians_degrees(self):
        # Ensure source + target units are handled, grid+result in degrees.
        for axis_name in ('x', 'y'):
            self.src_cube.coord(axis=axis_name).convert_units('radians')
            self.grid_cube.coord(axis=axis_name).convert_units('degrees')
        result = self._check_expected()
        self.assertEqual(result.coord(axis='x').units, 'degrees')

    def test_degrees_radians(self):
        # Ensure source + target units are handled, grid+result in radians.
        for axis_name in ('x', 'y'):
            self.src_cube.coord(axis=axis_name).convert_units('degrees')
            self.grid_cube.coord(axis=axis_name).convert_units('radians')
        result = self._check_expected()
        self.assertEqual(result.coord(axis='x').units, 'radians')


if __name__ == "__main__":
    tests.main()
