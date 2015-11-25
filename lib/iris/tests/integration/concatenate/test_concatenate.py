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
"""
Integration tests for concatenating cubes with differing time coord epochs
using :func:`iris.util.unify_time_units`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.coords import AuxCoord, DimCoord
from iris._concatenate import concatenate
from iris.exceptions import ConcatenateError
from iris.util import unify_time_units
from cf_units import Unit


class Test_concatenate__epoch(tests.IrisTest):
    def simple_1d_time_cubes(self, reftimes, coords_points):
        cubes = []
        data_points = [273, 275, 278, 277, 274]
        for reftime, coord_points in zip(reftimes, coords_points):
            cube = iris.cube.Cube(np.array(data_points, dtype=np.float32),
                                  standard_name='air_temperature',
                                  units='K')
            unit = Unit(reftime, calendar='gregorian')
            coord = DimCoord(points=np.array(coord_points,
                                             dtype=np.float32),
                             standard_name='time',
                             units=unit)
            cube.add_dim_coord(coord, 0)
            cubes.append(cube)
        return cubes

    def test_concat_1d_with_differing_time_units(self):
        reftimes = ['hours since 1970-01-01 00:00:00',
                    'hours since 1970-01-02 00:00:00']
        coords_points = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        cubes = self.simple_1d_time_cubes(reftimes, coords_points)
        unify_time_units(cubes)
        result = concatenate(cubes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10,))


class TestMessages__cube_signature(tests.IrisTest):
    def setUp(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        # Time coord
        t_unit = Unit('hours since 1970-01-01 00:00:00', calendar='gregorian')
        t_coord = DimCoord(points=np.arange(2, dtype=np.float32),
                           standard_name='time', units=t_unit)
        cube.add_dim_coord(t_coord, 0)
        # Lats and lons
        x_coord = DimCoord(points=np.arange(3, dtype=np.float32),
                           standard_name='longitude', units='degrees')
        cube.add_dim_coord(x_coord, 1)
        y_coord = DimCoord(points=np.arange(4, dtype=np.float32),
                           standard_name='latitude', units='degrees')
        cube.add_dim_coord(y_coord, 2)
        # Scalars
        cube.add_aux_coord(AuxCoord([0], "height", units="m"))
        # Aux Coords
        cube.add_aux_coord(AuxCoord(data, long_name='wibble', units='1'),
                           data_dims=(0, 1, 2))
        cube.add_aux_coord(AuxCoord([0, 1, 2], long_name='foo', units='1'),
                           data_dims=(1,))
        self.cube = cube

    def test_definition_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.units = '1'
        exc_regexp = 'Cube metadata differs for phenomenon: *'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_dimensions_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('latitude')
        exc_regexp = 'Dimension coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_dimensions_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('latitude').long_name = 'bob'
        exc_regexp = 'Dimension coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_aux_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('foo')
        exc_regexp = 'Auxiliary coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_aux_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('foo').units = 'm'
        exc_regexp = 'Auxiliary coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_scalar_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord('height')
        exc_regexp = 'Scalar coordinates differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_scalar_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord('height').long_name = 'alice'
        exc_regexp = 'Scalar coordinates metadata differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

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
            concatenate([cube_1, cube_2], True)

    def test_datatype_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.data.dtype = np.float64
        exc_regexp = 'Datatypes differ: .* != .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)


class TestMessages__coords_signature(tests.IrisTest):

    def setUp(self):
        self.cube_1 = self._build_cube(3, 2, 4)
        self.cube_2 = self._build_cube(3, 2, 4, t_offset=4)

    @staticmethod
    def _build_cube(t, lat, lon, t_offset=0):
        # Construct a 3D time-latitude-longitude cube.
        # Also includes a 3D aux coord `wibble` spanning all 3 dims,
        # and a 1D aux coord `foo` on dim 1.
        # `t_offset` specifies a scalar offset for time points, which allows
        # for construction of time coords that can be concatenated.
        data = np.arange(np.prod([t, lat, lon]),
                         dtype=np.float32).reshape(t, lon, lat)
        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        # Time coord.
        t_unit = Unit('hours since 1970-01-01 00:00:00', calendar='gregorian')
        t_points = np.arange(t, dtype=np.float32) + t_offset
        t_coord = DimCoord(points=t_points, standard_name='time', units=t_unit)
        cube.add_dim_coord(t_coord, 0)
        # Lat and lon coords.
        x_coord = DimCoord(points=np.arange(lon, dtype=np.float32),
                           standard_name='longitude', units='degrees')
        cube.add_dim_coord(x_coord, 1)
        y_coord = DimCoord(points=np.arange(lat, dtype=np.float32),
                           standard_name='latitude', units='degrees')
        cube.add_dim_coord(y_coord, 2)
        # Aux Coords.
        cube.add_aux_coord(AuxCoord(data, long_name='wibble', units='1'),
                           data_dims=(0, 1, 2))
        cube.add_aux_coord(AuxCoord(np.arange(lon),
                                    long_name='foo', units='1'),
                           data_dims=(1,))
        return cube

    # Candidate axis checks.
    def test_identical_cubes_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_1.copy()
        exc_regexp = 'Cubes are identical'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_candidate_coordinates_overlap_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_2.copy()
        cube_2.coord('time').points[0] = cube_1.coord('time').points[1]
        exc_regexp = 'Coordinates overlap'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_candidate_coordinates_not_monotonic_message(self):
        cube_1 = self.cube_1
        cube_2 = cube_1.copy()
        cube_2.coord('time').points = cube_1.coord('time').points + 1
        exc_regexp = 'not form a monotonic coordinate'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    # Other dim coords checks.
    def test_dim_coord_points_not_equal_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_2.copy()
        cube_2.coord('latitude').points[0] = 1000
        exc_regexp = '.* Coordinate points .* latitude .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_dim_coord_bounds_extents_not_equal_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_2.copy()
        cube_2.coord('latitude').guess_bounds()
        exc_regexp = 'Coordinate bounds .* latitude .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    # Aux coords checks.
    def test_aux_coord_points_not_equal_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_2.copy()
        cube_2.coord('foo').points[0] = 1000
        exc_regexp = 'Aux coordinate points .* foo .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_aux_coord_bounds_not_equal_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_2.copy()
        cube_2.coord('foo').guess_bounds()
        exc_regexp = 'Aux coordinate bounds .* foo .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    def test_aux_coord_covered_dims_not_equal_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_2.copy()
        cube_2.remove_coord('foo')
        new_points = np.arange(12).reshape(3, 4)
        cube_2.add_aux_coord(AuxCoord(new_points, long_name='foo', units='1'),
                             data_dims=(0, 1))
        exc_regexp = 'Covered dimensions .* foo .*'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)

    # Mismatched anonymous coords checks.
    def test_mismatched_anonymous_dims_message(self):
        cube_1 = self.cube_1
        cube_2 = self.cube_2.copy()
        cube_2.remove_coord('longitude')
        exc_regexp = 'Mismatched anonymous'
        with self.assertRaisesRegexp(ConcatenateError, exc_regexp):
            concatenate([cube_1, cube_2], True)


if __name__ == '__main__':
    tests.main()
