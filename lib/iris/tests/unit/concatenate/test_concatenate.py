# (C) British Crown Copyright 2014 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import biggus
import numpy as np

import iris.coords
from iris._concatenate import concatenate
import iris.cube
from iris.exceptions import ConcatenateError
import cf_units


class TestEpoch(tests.IrisTest):
    def simple_1d_time_cubes(self, reftimes, coords_points):
        cubes = []
        data_points = [273, 275, 278, 277, 274]
        for reftime, coord_points in zip(reftimes, coords_points):
            cube = iris.cube.Cube(np.array(data_points, dtype=np.float32),
                                  standard_name='air_temperature',
                                  units='K')
            unit = cf_units.Unit(reftime, calendar='gregorian')
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


class TestMessages(tests.IrisTest):
    def setUp(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        # Time coord
        t_unit = cf_units.Unit('hours since 1970-01-01 00:00:00',
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
            result = concatenate([cube_1, cube_2], True)

    def test_definition_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.units = '1'
        exc_regexp = 'Cube metadata differs for phenomenon: *'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)

    def test_dimensions_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('latitude')
        exc_regexp = 'Dimension coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)

    def test_dimensions_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('latitude').long_name = 'bob'
        exc_regexp = 'Dimension coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)

    def test_aux_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('foo')
        exc_regexp = 'Auxiliary coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)

    def test_aux_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('foo').units = 'm'
        exc_regexp = 'Auxiliary coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)

    def test_scalar_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('height')
        exc_regexp = 'Scalar coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)

    def test_scalar_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('height').long_name = 'alice'
        exc_regexp = 'Scalar coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)

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
            result = concatenate([cube_1, cube_2], True)

    def test_datatype_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.data.dtype = np.float64
        exc_regexp = 'Datatypes differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            result = concatenate([cube_1, cube_2], True)


class TestOrder(tests.IrisTest):
    def _make_cube(self, points, bounds=None):
        nx = 4
        data = np.arange(len(points) * nx).reshape(len(points), nx)
        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        lat = iris.coords.DimCoord(points, 'latitude', bounds=bounds)
        lon = iris.coords.DimCoord(np.arange(nx), 'longitude')
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        return cube

    def test_asc_points(self):
        top = self._make_cube([10, 30, 50, 70, 90])
        bottom = self._make_cube([-90, -70, -50, -30, -10])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_asc_bounds(self):
        top = self._make_cube([22.5, 67.5], [[0, 45], [45, 90]])
        bottom = self._make_cube([-67.5, -22.5], [[-90, -45], [-45, 0]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_points(self):
        top = self._make_cube([90, 70, 50, 30, 10])
        bottom = self._make_cube([-10, -30, -50, -70, -90])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_bounds(self):
        top = self._make_cube([67.5, 22.5], [[90, 45], [45, 0]])
        bottom = self._make_cube([-22.5, -67.5], [[0, -45], [-45, -90]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)


class TestConcatenateBiggus(tests.IrisTest):
    def build_lazy_cube(self, points, bounds=None, nx=4):
        data = np.arange(len(points) * nx).reshape(len(points), nx)
        data = biggus.NumpyArrayAdapter(data)
        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        lat = iris.coords.DimCoord(points, 'latitude', bounds=bounds)
        lon = iris.coords.DimCoord(np.arange(nx), 'longitude')
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        return cube

    def test_lazy_biggus_concatenate(self):
        c1 = self.build_lazy_cube([1, 2])
        c2 = self.build_lazy_cube([3, 4, 5])
        cube, = concatenate([c1, c2])
        self.assertTrue(cube.has_lazy_data())
        self.assertNotIsInstance(cube.data, np.ma.MaskedArray)

    def test_lazy_biggus_concatenate_masked_array_mixed_deffered(self):
        c1 = self.build_lazy_cube([1, 2])
        c2 = self.build_lazy_cube([3, 4, 5])
        c2.data = np.ma.masked_greater(c2.data, 3)
        self.assertFalse(c2.has_lazy_data())
        cube, = concatenate([c1, c2])
        self.assertTrue(cube.has_lazy_data())
        self.assertIsInstance(cube.data, np.ma.MaskedArray)


if __name__ == '__main__':
    tests.main()
