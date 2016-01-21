# (C) British Crown Copyright 2014 - 2016, Met Office
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

import cf_units
import numpy as np

import iris.coords
from iris._concatenate import concatenate
import iris.cube
from iris.util import unify_time_units


class Test_concatenate__epoch(tests.IrisTest):
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

    def test_concat_1d_with_differing_time_units(self):
        reftimes = ['hours since 1970-01-01 00:00:00',
                    'hours since 1970-01-02 00:00:00']
        coords_points = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        cubes = self.simple_1d_time_cubes(reftimes, coords_points)
        unify_time_units(cubes)
        result = concatenate(cubes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10,))


class Test_cubes_with_aux_coord(tests.IrisTest):
    def create_cube(self):
        data = np.arange(4).reshape(2, 2)

        lat = iris.coords.DimCoord([0, 30], standard_name='latitude',
                                   units='degrees')
        lon = iris.coords.DimCoord([0, 15], standard_name='longitude',
                                   units='degrees')
        height = iris.coords.AuxCoord([1.5], standard_name='height', units='m')
        t_unit = cf_units.Unit('hours since 1970-01-01 00:00:00',
                               calendar='gregorian')
        time = iris.coords.DimCoord([0, 6], standard_name='time', units=t_unit)

        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        cube.add_dim_coord(time, 0)
        cube.add_dim_coord(lat, 1)
        cube.add_aux_coord(lon, 1)
        cube.add_aux_coord(height)
        return cube

    def test_diff_aux_coord(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord('time').points = [12, 18]
        cube_b.coord('longitude').points = [120, 150]

        result = concatenate([cube_a, cube_b])
        self.assertEqual(len(result), 2)

    def test_ignore_diff_aux_coord(self):
        cube_a = self.create_cube()
        cube_b = cube_a.copy()
        cube_b.coord('time').points = [12, 18]
        cube_b.coord('longitude').points = [120, 150]

        result = concatenate([cube_a, cube_b], check_aux_coords=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (4, 2))


class Test_anonymous_dims(tests.IrisTest):
    def setUp(self):
        data = np.arange(12).reshape(2, 3, 2)
        self.cube = iris.cube.Cube(data, standard_name='air_temperature',
                                   units='K')

        # Time coord
        t_unit = cf_units.Unit('hours since 1970-01-01 00:00:00',
                               calendar='gregorian')
        t_coord = iris.coords.DimCoord([0, 6],
                                       standard_name='time',
                                       units=t_unit)
        self.cube.add_dim_coord(t_coord, 0)

        # Lats and lons
        self.x_coord = iris.coords.DimCoord([15, 30],
                                            standard_name='longitude',
                                            units='degrees')
        self.y_coord = iris.coords.DimCoord([0, 30, 60],
                                            standard_name='latitude',
                                            units='degrees')
        self.x_coord_2D = iris.coords.AuxCoord([[0, 15], [30, 45], [60, 75]],
                                               standard_name='longitude',
                                               units='degrees')
        self.y_coord_non_monotonic = iris.coords.AuxCoord(
            [0, 30, 15], standard_name='latitude', units='degrees')

    def test_matching_2d_longitudes(self):
        cube1 = self.cube
        cube1.add_dim_coord(self.y_coord, 1)
        cube1.add_aux_coord(self.x_coord_2D, (1, 2))

        cube2 = cube1.copy()
        cube2.coord('time').points = [12, 18]
        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 1)

    def test_differing_2d_longitudes(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord, 1)
        cube1.add_aux_coord(self.x_coord_2D, (1, 2))

        cube2 = cube1.copy()
        cube2.coord('time').points = [12, 18]
        cube2.coord('longitude').points = [[-30, -15], [0, 15], [30, 45]]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 2)

    def test_matching_non_monotonic_latitudes(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord_non_monotonic, 1)
        cube1.add_aux_coord(self.x_coord, 2)

        cube2 = cube1.copy()
        cube2.coord('time').points = [12, 18]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 1)

    def test_differing_non_monotonic_latitudes(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord_non_monotonic, 1)
        cube1.add_aux_coord(self.x_coord, 2)

        cube2 = cube1.copy()
        cube2.coord('time').points = [12, 18]
        cube2.coord('latitude').points = [30, 0, 15]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 2)

    def test_concatenate_anonymous_dim(self):
        cube1 = self.cube
        cube1.add_aux_coord(self.y_coord_non_monotonic, 1)
        cube1.add_aux_coord(self.x_coord, 2)

        cube2 = cube1.copy()
        cube2.coord('latitude').points = [30, 0, 15]

        result = concatenate([cube1, cube2])
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    tests.main()
