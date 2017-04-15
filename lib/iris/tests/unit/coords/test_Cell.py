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
import netcdftime
import numpy as np

from iris.coords import Cell
from iris.time import PartialDateTime


class Test___common_cmp__(tests.IrisTest):
    def assert_raises_on_comparison(self, cell, other, exception_type, regexp):
        with self.assertRaisesRegexp(exception_type, regexp):
            cell < other
        with self.assertRaisesRegexp(exception_type, regexp):
            cell <= other
        with self.assertRaisesRegexp(exception_type, regexp):
            cell > other
        with self.assertRaisesRegexp(exception_type, regexp):
            cell >= other

    def test_netcdftime_cell(self):
        # Check that cell comparison when the cell contains
        # netcdftime.datetime objects raises an exception otherwise
        # this will fall back to id comparison producing unreliable
        # results.
        cell = Cell(netcdftime.datetime(2010, 3, 21))
        dt = mock.Mock(timetuple=mock.Mock())
        self.assert_raises_on_comparison(cell, dt, TypeError,
                                         'determine the order of netcdftime')
        self.assert_raises_on_comparison(cell, 23, TypeError,
                                         'determine the order of netcdftime')
        self.assert_raises_on_comparison(cell, 'hello', TypeError,
                                         'Unexpected type.*str')

    def test_netcdftime_other(self):
        # Check that cell comparison to a netcdftime.datetime object
        # raises an exception otherwise this will fall back to id comparison
        # producing unreliable results.
        dt = netcdftime.datetime(2010, 3, 21)
        cell = Cell(mock.Mock(timetuple=mock.Mock()))
        self.assert_raises_on_comparison(cell, dt, TypeError,
                                         'determine the order of netcdftime')

    def test_PartialDateTime_bounded_cell(self):
        # Check that bounded comparisions to a PartialDateTime
        # raise an exception. These are not supported as they
        # depend on the calendar.
        dt = PartialDateTime(month=6)
        cell = Cell(datetime.datetime(2010, 1, 1),
                    bound=[datetime.datetime(2010, 1, 1),
                           datetime.datetime(2011, 1, 1)])
        self.assert_raises_on_comparison(cell, dt, TypeError,
                                         'bounded region for datetime')

    def test_PartialDateTime_unbounded_cell(self):
        # Check that cell comparison works with PartialDateTimes.
        dt = PartialDateTime(month=6)
        cell = Cell(netcdftime.datetime(2010, 3, 1))
        self.assertLess(cell, dt)
        self.assertGreater(dt, cell)
        self.assertLessEqual(cell, dt)
        self.assertGreaterEqual(dt, cell)

    def test_datetime_unbounded_cell(self):
        # Check that cell comparison works with datetimes.
        dt = datetime.datetime(2000, 6, 15)
        cell = Cell(datetime.datetime(2000, 1, 1))
        # Note the absence of the inverse of these
        # e.g. self.assertGreater(dt, cell).
        # See http://bugs.python.org/issue8005
        self.assertLess(cell, dt)
        self.assertLessEqual(cell, dt)


class Test___eq__(tests.IrisTest):
    def test_datetimelike(self):
        # Check that cell equality works with objects with a "timetuple".
        dt = mock.Mock(timetuple=mock.Mock())
        cell = mock.MagicMock(spec=Cell, point=datetime.datetime(2010, 3, 21),
                              bound=None)
        _ = cell == dt
        cell.__eq__.assert_called_once_with(dt)

    def test_datetimelike_bounded_cell(self):
        # Check that equality with a datetime-like bounded cell
        # raises an error. This is not supported as it
        # depends on the calendar which is not always known from
        # the datetime-like bound objects.
        other = mock.Mock(timetuple=mock.Mock())
        cell = Cell(point=object(),
                    bound=[mock.Mock(timetuple=mock.Mock()),
                           mock.Mock(timetuple=mock.Mock())])
        with self.assertRaisesRegexp(TypeError, 'bounded region for datetime'):
            cell == other

    def test_PartialDateTime_other(self):
        cell = Cell(datetime.datetime(2010, 3, 2))
        # A few simple cases.
        self.assertEqual(cell, PartialDateTime(month=3))
        self.assertNotEqual(cell, PartialDateTime(month=3, hour=12))
        self.assertNotEqual(cell, PartialDateTime(month=4))


class Test_contains_point(tests.IrisTest):
    def test_datetimelike_bounded_cell(self):
        point = object()
        cell = Cell(point=object(),
                    bound=[mock.Mock(timetuple=mock.Mock()),
                           mock.Mock(timetuple=mock.Mock())])
        with self.assertRaisesRegexp(TypeError, 'bounded region for datetime'):
            cell.contains_point(point)

    def test_datetimelike_point(self):
        point = mock.Mock(timetuple=mock.Mock())
        cell = Cell(point=object(), bound=[object(), object()])
        with self.assertRaisesRegexp(TypeError, 'bounded region for datetime'):
            cell.contains_point(point)


class Test_contains_point__PartialDateTime(tests.IrisTest):
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
