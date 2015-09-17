# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for the `iris.cube.Cube` class operators."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import iris
import iris.tests as tests
import numpy as np
import biggus


class Test_Lazy_Maths(tests.IrisTest):
    def build_lazy_cube(self, points, bounds=None, nx=10):
        data = np.arange(len(points) * nx).reshape(len(points), nx)
        data = biggus.NumpyArrayAdapter(data)
        cube = iris.cube.Cube(data, standard_name='air_temperature', units='K')
        lat = iris.coords.DimCoord(points, 'latitude', bounds=bounds)
        lon = iris.coords.DimCoord(np.arange(nx), 'longitude')
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        return cube

    def assert_elementwise(self, cube, other, result, np_op):
        self.assertIsInstance(result, biggus._Elementwise)
        self.assertEqual(result._numpy_op, np_op)
        self.assertArrayAlmostEqual(result._array1, cube.lazy_data())
        if other is not None:
            self.assertArrayAlmostEqual(result._array2, other)

    def test_lazy_biggus_add_cubes(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 + c1
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, c1.lazy_data(), result, np.add)

    def test_lazy_biggus_add_scalar(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 + 5
        self.assertEqual(c1 + 5, 5 + c1)
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, None, result, np.add)

    def test_lazy_biggus_mul_cubes(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 * c1
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, c1.lazy_data(), result, np.multiply)

    def test_lazy_biggus_mul_scalar(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 * 5
        self.assertEqual(c1 * 5, 5 * c1)
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, None, result, np.multiply)

    def test_lazy_biggus_sub_cubes(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 - c1
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, c1.lazy_data(), result, np.subtract)

    def test_lazy_biggus_sub_scalar(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 - 5
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, None, result, np.subtract)

    def test_lazy_biggus_div_cubes(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 / c1
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, c1.lazy_data(), result, np.divide)

    def test_lazy_biggus_div_scalar(self):
        c1 = self.build_lazy_cube([1, 2])
        cube = c1 / 5
        result = cube.lazy_data()
        self.assertTrue(cube.has_lazy_data())
        self.assert_elementwise(c1, None, result, np.divide)


if __name__ == "__main__":
    tests.main()
