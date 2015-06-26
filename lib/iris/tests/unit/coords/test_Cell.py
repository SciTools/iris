# (C) British Crown Copyright 2013 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import datetime

import netcdftime

from iris.coords import Cell
from iris.tests import mock
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


if __name__ == '__main__':
    tests.main()
