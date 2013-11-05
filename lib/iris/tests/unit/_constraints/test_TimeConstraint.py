# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris._constraints.TimeConstraint` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris._constraints import TimeConstraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError


class Test___init__(tests.IrisTest):
    def test_default(self):
        constraint = TimeConstraint()
        self.assertIsNone(constraint.hour)
        self.assertEqual(constraint.coord, 'time')

    def test_explicit(self):
        constraint = TimeConstraint(hour=4, coord='quee')
        self.assertEqual(constraint.hour, 4)
        self.assertEqual(constraint.coord, 'quee')


class Test___init____invalid_hour(tests.IrisTest):
    def test_negative(self):
        with self.assertRaises(ValueError):
            constraint = TimeConstraint(hour=-3)

    def test_too_large(self):
        with self.assertRaises(ValueError):
            constraint = TimeConstraint(hour=24)

    def test_wrong_type(self):
        with self.assertRaises(ValueError):
            constraint = TimeConstraint(hour='quangle')


class Test__repr__(tests.IrisTest):
    def test(self):
        constraint = TimeConstraint(hour=9)
        self.assertEqual(repr(constraint), 'TimeConstraint(hour=9)')


class Test_extract__hour_scalar_coord(tests.IrisTest):
    def _cube(self, point_value):
        cube = Cube(np.arange(5 * 6).reshape(5, 6))
        cube.add_aux_coord(AuxCoord(point_value, 'time',
                                    units='hours since 2013-10-29 18:00:00'))
        return cube

    def test_scalar_time_coord_match(self):
        cube = self._cube(18)
        constraint = TimeConstraint(hour=12)
        sub_cube = constraint.extract(cube)
        self.assertIs(sub_cube, cube)

    def test_scalar_time_coord_no_match(self):
        cube = self._cube(0)
        constraint = TimeConstraint(hour=12)
        sub_cube = constraint.extract(cube)
        self.assertIsNone(sub_cube)


class Test_extract__hour_points(tests.IrisTest):
    def _1d_cube(self):
        time = DimCoord(np.arange(12) * 6, 'time',
                        units='hours since 2013-10-29 18:00:00')
        cube = Cube(np.arange(12))
        cube.add_dim_coord(time, 0)
        return cube

    def test_1d_data(self):
        cube = self._1d_cube()
        constraint = TimeConstraint(hour=12)
        sub_cube = constraint.extract(cube)
        self.assertArrayEqual(sub_cube.coord('time').points, [18, 42, 66])

    def test_3d_data(self):
        time = DimCoord(np.arange(12) * 6, 'time',
                        units='hours since 2013-10-29 18:00:00')
        cube = Cube(np.arange(12 * 5 * 6).reshape(12, 5, 6))
        cube.add_dim_coord(time, 0)
        constraint = TimeConstraint(hour=12)
        sub_cube = constraint.extract(cube)
        self.assertArrayEqual(sub_cube.coord('time').points, [18, 42, 66])

    def test_no_hour_match(self):
        cube = self._1d_cube()
        constraint = TimeConstraint(hour=13)
        sub_cube = constraint.extract(cube)
        self.assertIsNone(sub_cube)

    def test_default(self):
        cube = self._1d_cube()
        constraint = TimeConstraint()
        sub_cube = constraint.extract(cube)
        self.assertIs(sub_cube, cube)

    def test_none(self):
        cube = self._1d_cube()
        constraint = TimeConstraint(hour=None)
        sub_cube = constraint.extract(cube)
        self.assertIs(sub_cube, cube)


class Test_extract__alternative_coord(tests.IrisTest):
    def _1d_cube(self):
        time = DimCoord(np.arange(12) * 6, long_name='wangle',
                        units='hours since 2013-10-29 18:00:00')
        cube = Cube(np.arange(12))
        cube.add_dim_coord(time, 0)
        return cube

    def test_1d(self):
        cube = self._1d_cube()
        constraint = TimeConstraint(hour=12, coord='wangle')
        sub_cube = constraint.extract(cube)
        self.assertArrayEqual(sub_cube.coord('wangle').points, [18, 42, 66])

    def test_no_coord_match(self):
        cube = self._1d_cube()
        constraint = TimeConstraint(hour=12)
        sub_cube = constraint.extract(cube)
        self.assertIsNone(sub_cube)


class Test_extract__invalid(tests.IrisTest):
    def test_multi_dimensional(self):
        time = AuxCoord(np.arange(12).reshape(3, 4), 'time',
                        units='hours since 2013-10-29 18:00:00')
        cube = Cube(np.arange(12).reshape(3, 4))
        cube.add_aux_coord(time, (0, 1))
        constraint = TimeConstraint(hour=12)
        with self.assertRaises(CoordinateMultiDimError):
            sub_cube = constraint.extract(cube)


if __name__ == "__main__":
    tests.main()
