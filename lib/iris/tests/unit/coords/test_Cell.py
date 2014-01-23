# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the :class:`iris.coords.Cell` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import datetime

import mock

from iris.coords import Cell
from iris.time import PartialDateTime


class Test___common_cmp__(tests.IrisTest):
    def test_datetime_ordering(self):
        # Check that cell comparison works with objects with a "timetuple".
        dt = mock.Mock(timetuple=mock.Mock())
        cell = Cell(datetime.datetime(2010, 3, 21))
        with mock.patch('operator.gt') as gt:
            _ = cell > dt
        gt.assert_called_once_with(cell.point, dt)

        # Now check that the existence of timetuple is causing that.
        del dt.timetuple
        with self.assertRaisesRegexp(ValueError,
                                     'Unexpected type of other <(.*)>'):
            _ = cell > dt

    def test_datetime_equality(self):
        # Check that cell equality works with objects with a "timetuple".
        dt = mock.Mock(timetuple=mock.Mock())
        cell = mock.MagicMock(spec=Cell, point=datetime.datetime(2010, 3, 21),
                              bound=None)
        _ = cell == dt
        cell.__eq__.assert_called_once_with(dt)


class Test_contains_point(tests.IrisTest):
    def test_cross_end_of_year_month(self):
        cell = Cell(datetime.datetime(2012, 2, 1),
                    [datetime.datetime(2011, 12, 1),
                     datetime.datetime(2012, 3, 1)])
        for month in (12, 1, 2, 3):
            self.assertTrue(cell.contains_point(PartialDateTime(month=month)))
        for month in range(4, 12):
            self.assertFalse(cell.contains_point(PartialDateTime(month=month)))

    def test_within_year_month(self):
        cell = Cell(datetime.datetime(2012, 7, 1),
                    [datetime.datetime(2012, 6, 15),
                     datetime.datetime(2012, 9, 10)])
        for month in range(6, 10):
            self.assertTrue(cell.contains_point(PartialDateTime(month=month)))
        for month in range(1, 6) + range(11, 13):
            self.assertFalse(cell.contains_point(PartialDateTime(month=month)))

    def test_within_year_month_descending(self):
        cell = Cell(datetime.datetime(2012, 7, 1),
                    [datetime.datetime(2012, 9, 10),
                     datetime.datetime(2012, 6, 15)])
        for month in range(6, 10):
            self.assertTrue(cell.contains_point(PartialDateTime(month=month)))
        for month in range(1, 6) + range(11, 13):
            self.assertFalse(cell.contains_point(PartialDateTime(month=month)))

    def test_cross_end_of_year_day(self):
        cell = Cell(datetime.datetime(2012, 1, 1),
                    [datetime.datetime(2011, 12, 15),
                     datetime.datetime(2012, 1, 4)])
        self.assertTrue(cell.contains_point(PartialDateTime(day=20)))
        for day in range(1, 5) + range(16, 32):
            self.assertTrue(cell.contains_point(PartialDateTime(day=day)))
        for day in range(5, 15):
            self.assertFalse(cell.contains_point(PartialDateTime(day=day)))

    def test_year(self):
        cell = Cell(datetime.datetime(2000, 1, 1),
                    [datetime.datetime(1990, 1, 1),
                     datetime.datetime(2010, 1, 1)])
        for year in range(1990, 2011):
            self.assertTrue(cell.contains_point(PartialDateTime(year=year)))
        for year in (1989, 2011):
            self.assertFalse(cell.contains_point(PartialDateTime(year=year)))

    def test_cross_end_of_year_month_and_day(self):
        cell = Cell(datetime.datetime(2012, 1, 1),
                    [datetime.datetime(2011, 12, 15),
                     datetime.datetime(2012, 3, 4)])

        in_combos = ((12, range(15, 32)),
                     (1, range(1, 32)),
                     (2, range(1, 29)),
                     (3, range(1, 5)))
        for month, days in in_combos:
            for day in days:
                self.assertTrue(
                    cell.contains_point(PartialDateTime(month=month, day=day)))
        out_combos = ((3, range(5, 32)),
                      (4, range(1, 31)),
                      (5, range(1, 32)),
                      (6, range(1, 31)),
                      (7, range(1, 32)),
                      (8, range(1, 32)),
                      (9, range(1, 31)),
                      (10, range(1, 32)),
                      (11, range(1, 31)),
                      (12, range(1, 15)))
        for month, days in out_combos:
            for day in days:
                self.assertFalse(
                    cell.contains_point(PartialDateTime(month=month, day=day)))


if __name__ == '__main__':
    tests.main()
