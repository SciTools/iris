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
"""Integration tests for constraint handling of the time coordinate."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.pdatetime import PartialDateTime


class TestTimeConstraint_basic(tests.IrisTest):
    def setUp(self):
        # ([1970-01-01 00:00:00, 1970-07-01 00:00:00,
        #   1971-01-01 00:00:00, 1971-07-01 00:00:00],
        #  standard_name='time', calendar='360_day')
        self.cube = iris.cube.Cube(np.arange(4))

        time = iris.coords.DimCoord(
            np.arange(4) * 4320, "time",
            units=iris.unit.Unit("hours since epoch", calendar='360_day'))
        self.cube.add_dim_coord(time, 0)

    def test_time_with_number(self):
        # Providing traditional numeric value as constraint.
        constraint = iris.Constraint(time=8640)
        cube = self.cube.extract(constraint)
        com_coord = self.cube.coord('time')[2]
        self.assertEqual(cube.coord('time'), com_coord)

    def test_time_with_partial(self):
        # Providing the new partial datetime object.
        constraint = iris.Constraint(time=PartialDateTime(year=1971, month=1))
        cube = self.cube.extract(constraint)
        com_coord = self.cube.coord('time')[2]
        self.assertEqual(cube.coord('time'), com_coord)

    def test_lambda_constraint(self):
        constraint = iris.Constraint(
            time=lambda cell: cell.point.year == 1970 and
            cell.point.month == 1)
        cube = self.cube.extract(constraint)
        com_coord = self.cube.coord('time')[0]
        self.assertEqual(cube.coord('time'), com_coord)

    def test_lambda_constraint_alt(self):
        constraint = iris.Constraint(
            time=lambda cell: cell.point == PartialDateTime(year=1970,
                                                            month=1))
        cube = self.cube.extract(constraint)
        com_coord = self.cube.coord('time')[0]
        self.assertEqual(cube.coord('time'), com_coord)

    def test_extract_range_point(self):
        time_lo = PartialDateTime(year=1970, month=1)
        time_hi = PartialDateTime(year=1971, month=1)
        cube = self.cube.extract(iris.Constraint(
            time=lambda cell: time_lo <= cell.point <= time_hi))
        com_coord = self.cube.coord('time')[:-1]
        self.assertEqual(cube.coord('time'), com_coord)

    def test_extract_range_cell(self):
        time_lo = PartialDateTime(year=1970, month=1)
        time_hi = PartialDateTime(year=1971, month=1)
        cube = self.cube.extract(iris.Constraint(
            time=lambda cell: time_lo <= cell <= time_hi))
        com_coord = self.cube.coord('time')[:-1]
        self.assertEqual(cube.coord('time'), com_coord)


if __name__ == "__main__":
    tests.main()
