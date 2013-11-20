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
"""Unit tests for `iris.experimental.constraints.TimeRangeConstraint`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.experimental.constraints import TimeRangeConstraint
from iris.coords import AuxCoord, DimCoord
import iris.cube
import iris.unit


def _point_cube(point, cal):
    cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
    units = iris.unit.Unit('days since 2001-01-01 00:00:00', calendar=cal)
    cube.add_aux_coord(AuxCoord(point, 'time', units=units))
    return cube


def _point_cube_leap(point, cal):
    cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
    units = iris.unit.Unit('days since 2000-01-01 00:00:00', calendar=cal)
    cube.add_aux_coord(AuxCoord(point, 'time', units=units))
    return cube


def _vector_cube(cal):
    cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
    units = iris.unit.Unit('days since 2001-01-01 00:00:00', calendar=cal)
    cube.add_dim_coord(DimCoord(np.arange(6) * 60, 'time', units=units), 1)
    return cube


def _vector_cube_leap(cal):
    cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
    units = iris.unit.Unit('days since 2000-01-01 00:00:00', calendar=cal)
    cube.add_dim_coord(DimCoord(np.arange(6) * 60, 'time', units=units), 1)
    return cube


class Test_extract__from_point(tests.IrisTest):

    constraint = TimeRangeConstraint(day_of_year=[(3, 1), (9, 30)])

    def test_greg_in(self):
        cube = _point_cube(59, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, cube)

    def test_greg_out(self):
        cube = _point_cube(58, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, None)

    def test_greg_leap_in(self):
        cube = _point_cube_leap(60, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, cube)

    def test_greg_leap_out(self):
        cube = _point_cube_leap(59, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, None)

    def test_360_in(self):
        cube = _point_cube(60, iris.unit.CALENDAR_360_DAY)
        result = self.constraint.extract(cube)
        self.assertEqual(result, cube)

    def test_360_out(self):
        cube = _point_cube(59, iris.unit.CALENDAR_360_DAY)
        result = self.constraint.extract(cube)
        self.assertEqual(result, None)


class Test_extract__from_point_wrap(tests.IrisTest):

    constraint = TimeRangeConstraint(day_of_year=[(9, 30), (3, 1)])

    def _cube(self, point, cal):
        cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
        units = iris.unit.Unit('days since 2001-01-01 00:00:00', calendar=cal)
        cube.add_aux_coord(AuxCoord(point, 'time', units=units))
        return cube

    def _cube_leap(self, point, cal):
        cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
        units = iris.unit.Unit('days since 2000-01-01 00:00:00', calendar=cal)
        cube.add_aux_coord(AuxCoord(point, 'time', units=units))
        return cube

    def test_greg_in(self):
        cube = _point_cube(59, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, cube)

    def test_greg_out(self):
        cube = _point_cube(60, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, None)

    def test_greg_leap_in(self):
        cube = _point_cube_leap(60, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, cube)

    def test_greg_leap_out(self):
        cube = _point_cube_leap(61, iris.unit.CALENDAR_GREGORIAN)
        result = self.constraint.extract(cube)
        self.assertEqual(result, None)

    def test_360_in(self):
        cube = _point_cube(60, iris.unit.CALENDAR_360_DAY)
        result = self.constraint.extract(cube)
        self.assertEqual(result, cube)

    def test_360_out(self):
        cube = _point_cube(61, iris.unit.CALENDAR_360_DAY)
        result = self.constraint.extract(cube)
        self.assertEqual(result, None)


class Test_extract__from_vector(tests.IrisTest):

    def test_greg(self):
        cube = _vector_cube(iris.unit.CALENDAR_GREGORIAN)
        constraint = TimeRangeConstraint(day_of_year=[(3, 2), (8, 29)])
        result = constraint.extract(cube)
        self.assertArrayEqual(result.coord('time').points, [60, 120, 180, 240])

    def test_greg_leap_in(self):
        cube = _vector_cube_leap(iris.unit.CALENDAR_GREGORIAN)
        constraint = TimeRangeConstraint(day_of_year=[(3, 1), (8, 28)])
        result = constraint.extract(cube)
        self.assertArrayEqual(result.coord('time').points, [60, 120, 180, 240])

    def test_360_in(self):
        cube = _vector_cube(iris.unit.CALENDAR_360_DAY)
        constraint = TimeRangeConstraint(day_of_year=[(3, 1), (9, 1)])
        result = constraint.extract(cube)
        self.assertArrayEqual(result.coord('time').points, [60, 120, 180, 240])


class Test_extract__from_vector_wrap(tests.IrisTest):

    def _cube(self, cal):
        cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
        units = iris.unit.Unit('days since 2001-01-01 00:00:00', calendar=cal)
        cube.add_dim_coord(DimCoord(np.arange(6) * 60, 'time', units=units), 1)

    def _cube_leap(self, cal):
        cube = iris.cube.Cube(np.arange(5 * 6).reshape(5, 6))
        units = iris.unit.Unit('days since 2000-01-01 00:00:00', calendar=cal)
        cube.add_dim_coord(DimCoord(np.arange(6) * 60, 'time', units=units), 1)
        return cube

    def test_greg(self):
        cube = _vector_cube(iris.unit.CALENDAR_GREGORIAN)
        constraint = TimeRangeConstraint(day_of_year=[(8, 29), (3, 2)])
        result = constraint.extract(cube)
        self.assertArrayEqual(result.coord('time').points, [0, 60, 240, 300])

    def test_greg_leap(self):
        cube = _vector_cube_leap(iris.unit.CALENDAR_GREGORIAN)
        constraint = TimeRangeConstraint(day_of_year=[(8, 28), (3, 1)])
        result = constraint.extract(cube)
        self.assertArrayEqual(result.coord('time').points, [0, 60, 240, 300])

    def test_360(self):
        cube = _vector_cube_leap(iris.unit.CALENDAR_360_DAY)
        constraint = TimeRangeConstraint(day_of_year=[(9, 1), (3, 1)])
        result = constraint.extract(cube)
        self.assertArrayEqual(result.coord('time').points, [0, 60, 240, 300])


class Test_extract__no_coord(tests.IrisTest):
    def test_no_coord(self):
        cube = _point_cube(59, iris.unit.CALENDAR_GREGORIAN)
        constraint = TimeRangeConstraint(day_of_year=[(3, 1), (9, 30)])
        cube.remove_coord('time')
        with self.assertRaises(AttributeError):
            result = self.constraint.extract(cube)


if __name__ == "__main__":
    tests.main()
