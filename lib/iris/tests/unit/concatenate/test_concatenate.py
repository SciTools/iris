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
