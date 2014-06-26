# (C) British Crown Copyright 2014, Met Office
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
"""Test function :func:`iris._concatenate.concatenate.py`."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import iris.coords
from iris._concatenate import concatenate
import iris.cube
from iris.cube import CubeList
from iris.exceptions import ConcatenateError
import iris.unit


class Test_concatenate__epoch(tests.IrisTest):
    def simple_1d_time_cubes(self, reftimes, coords_points):
        cubes = []
        data_points = [273, 275, 278, 277, 274]
        for reftime, coord_points in zip(reftimes, coords_points):
            cube = iris.cube.Cube(np.array(data_points, dtype=np.float32),
                                  standard_name='air_temperature',
                                  units='K')
            unit = iris.unit.Unit(reftime, calendar='gregorian')
            coord = iris.coords.DimCoord(points=np.array(coord_points,
                                                         dtype=np.float32),
                                         standard_name='time',
                                         units=unit)
            cube.add_dim_coord(coord, 0)
            cubes.append(cube)
        return cubes

    def test_concat_1d_with_same_time_units(self):
        reftimes = ['hours since 1970-01-01 00:00:00',
                    'hours since 1970-01-01 00:00:00']
        coords_points = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        cubes = self.simple_1d_time_cubes(reftimes, coords_points)
        result = concatenate(cubes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10,))


class Test_concatenate_messages(tests.IrisTest):
    def setUp(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        # Time coord
        t_unit = iris.unit.Unit('hours since 1970-01-01 00:00:00',
                                calendar='gregorian')
        t_coord = iris.coords.DimCoord(points=np.arange(2, dtype=np.float32),
                                       standard_name='time',
                                       units=t_unit)
        cube.add_dim_coord(t_coord, 0)
        # Lats and lons
        x_coord = iris.coords.DimCoord(points=np.arange(3, dtype=np.float32),
                                       standard_name='longitude',
                                       units='degrees')
        cube.add_dim_coord(x_coord, 1)
        y_coord = iris.coords.DimCoord(points=np.arange(4, dtype=np.float32),
                                       standard_name='latitude',
                                       units='degrees')
        cube.add_dim_coord(y_coord, 2)
        # Scalars
        cube.add_aux_coord(iris.coords.AuxCoord([0], "height", units="m"))
        # Aux Coords
        cube.add_aux_coord(iris.coords.AuxCoord(data,
                                                long_name='wibble',
                                                units='1'),
                           data_dims=(0, 1, 2))
        cube.add_aux_coord(iris.coords.AuxCoord([0, 1, 2],
                                                long_name='foo',
                                                units='1'),
                           data_dims=(1,))
        self.cube = cube

    def test_anonymous_coord_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('latitude')
        exc_regexp = 'one or both cubes have anonymous dimensions'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_definition_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.units = '1'
        exc_regexp = 'Cube metadata differs for phenomenon: *'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_dimensions_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('latitude')
        exc_regexp = 'Dimension coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_dimensions_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('latitude').long_name = 'bob'
        exc_regexp = 'Dimension coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_aux_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('foo')
        exc_regexp = 'Auxiliary coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_aux_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('foo').units = 'm'
        exc_regexp = 'Auxiliary coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_scalar_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('height')
        exc_regexp = 'Scalar coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_scalar_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('height').long_name = 'alice'
        exc_regexp = 'Scalar coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_ndim_difference_message(self):
        cube_1 = self.cube
        cube_2 = iris.cube.Cube(np.arange(5, dtype=np.float32),
                                standard_name='air_temperature',
                                units='K')
        x_coord = iris.coords.DimCoord(points=np.arange(5, dtype=np.float32),
                                       standard_name='longitude',
                                       units='degrees')
        cube_2.add_dim_coord(x_coord, 0)
        exc_regexp = 'Data dimensions differ: [0-9] != [0-9]'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

    def test_datatype_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.data.dtype = np.float64
        exc_regexp = 'Datatypes differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            CubeList([cube_1, cube_2]).concatenate_cube()

if __name__ == '__main__':
    tests.main()
