# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coords.Cell` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import datetime
from unittest import mock

import cftime
import numpy as np

from iris.coords import Cell
from iris.time import PartialDateTime


class Test___common_cmp__(tests.IrisTest):
    def assert_raises_on_comparison(self, cell, other, exception_type, regexp):
        with self.assertRaisesRegex(exception_type, regexp):
            cell < other
        with self.assertRaisesRegex(exception_type, regexp):
            cell <= other
        with self.assertRaisesRegex(exception_type, regexp):
            cell > other
        with self.assertRaisesRegex(exception_type, regexp):
            cell >= other

    def test_PartialDateTime_bounded_cell(self):
        # Check that bounded comparisons to a PartialDateTime
        # raise an exception. These are not supported as they
        # depend on the calendar.
        dt = PartialDateTime(month=6)
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        self.assert_raises_on_comparison(
            cell, dt, TypeError, "bounded region for datetime"
        )

    def test_PartialDateTime_unbounded_cell(self):
        # Check that cell comparison works with PartialDateTimes.
        dt = PartialDateTime(month=6)
        cell = Cell(cftime.datetime(2010, 3, 1))
        self.assertLess(cell, dt)
        self.assertGreater(dt, cell)
        self.assertLessEqual(cell, dt)
        self.assertGreaterEqual(dt, cell)

    def test_datetime_unbounded_cell(self):
        # Check that cell comparison works with datetimes.
        dt = datetime.datetime(2000, 6, 15)
        cell = Cell(cftime.datetime(2000, 1, 1))
        self.assertGreater(dt, cell)
        self.assertGreaterEqual(dt, cell)
        self.assertLess(cell, dt)
        self.assertLessEqual(cell, dt)

    def test_0D_numpy_array(self):
        # Check that cell comparison works with 0D numpy arrays

        cell = Cell(1.3)

        self.assertGreater(np.array(1.5), cell)
        self.assertLess(np.array(1.1), cell)
        self.assertGreaterEqual(np.array(1.3), cell)
        self.assertLessEqual(np.array(1.3), cell)

    def test_len_1_numpy_array(self):
        # Check that cell comparison works with numpy arrays of len=1

        cell = Cell(1.3)

        self.assertGreater(np.array([1.5]), cell)
        self.assertLess(np.array([1.1]), cell)
        self.assertGreaterEqual(np.array([1.3]), cell)
        self.assertLessEqual(np.array([1.3]), cell)


class Test___eq__(tests.IrisTest):
    def test_datetimelike(self):
        # Check that cell equality works with objects with a "timetuple".
        dt = mock.Mock(timetuple=mock.Mock())
        cell = mock.MagicMock(
            spec=Cell, point=datetime.datetime(2010, 3, 21), bound=None
        )
        _ = cell == dt
        cell.__eq__.assert_called_once_with(dt)

    def test_datetimelike_bounded_cell(self):
        # Check that equality with a datetime-like bounded cell
        # raises an error. This is not supported as it
        # depends on the calendar which is not always known from
        # the datetime-like bound objects.
        other = mock.Mock(timetuple=mock.Mock())
        cell = Cell(
            point=object(),
            bound=[
                mock.Mock(timetuple=mock.Mock()),
                mock.Mock(timetuple=mock.Mock()),
            ],
        )
        with self.assertRaisesRegex(TypeError, "bounded region for datetime"):
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
        cell = Cell(
            point=object(),
            bound=[
                mock.Mock(timetuple=mock.Mock()),
                mock.Mock(timetuple=mock.Mock()),
            ],
        )
        with self.assertRaisesRegex(TypeError, "bounded region for datetime"):
            cell.contains_point(point)

    def test_datetimelike_point(self):
        point = mock.Mock(timetuple=mock.Mock())
        cell = Cell(point=object(), bound=[object(), object()])
        with self.assertRaisesRegex(TypeError, "bounded region for datetime"):
            cell.contains_point(point)


class Test_numpy_comparison(tests.IrisTest):
    """
    Unit tests to check that the results of comparisons with numpy types can be
    used as truth values."""

    def test_cell_lhs(self):
        cell = Cell(point=1.5)
        n = np.float64(1.2)

        try:
            bool(cell < n)
            bool(cell <= n)
            bool(cell > n)
            bool(cell >= n)
            bool(cell == n)
            bool(cell != n)
        except:  # noqa
            self.fail(
                "Result of comparison could not be used as a truth value"
            )

    def test_cell_rhs(self):
        cell = Cell(point=1.5)
        n = np.float64(1.2)

        try:
            bool(n < cell)
            bool(n <= cell)
            bool(n > cell)
            bool(n >= cell)
            bool(n == cell)
            bool(n != cell)
        except:  # noqa
            self.fail(
                "Result of comparison could not be used as a truth value"
            )


if __name__ == "__main__":
    tests.main()
