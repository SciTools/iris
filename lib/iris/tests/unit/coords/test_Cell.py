# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coords.Cell` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import datetime

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
        # Check bounded cell comparisons to a PartialDateTime
        dt = PartialDateTime(month=6)
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        self.assertGreater(dt, cell)
        self.assertGreaterEqual(dt, cell)
        self.assertLess(cell, dt)
        self.assertLessEqual(cell, dt)

    def test_cftime_calender_bounded_cell(self):
        # Check that cell comparisons fail with different calendars
        dt = cftime.datetime(2010, 3, 1, calendar="360_day")
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        self.assert_raises_on_comparison(cell, dt, TypeError, "different calendars")

    def test_PartialDateTime_unbounded_cell(self):
        # Check that cell comparison works with PartialDateTimes.
        dt = PartialDateTime(month=6)
        cell = Cell(cftime.datetime(2010, 3, 1))
        self.assertLess(cell, dt)
        self.assertGreater(dt, cell)
        self.assertLessEqual(cell, dt)
        self.assertGreaterEqual(dt, cell)

    def test_datetime_unbounded_cell(self):
        # Check that cell comparison works with datetimes & cftimes.
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
        # Check that cell equality works with different datetime objects
        # using the same calendar
        point = cftime.datetime(2010, 1, 1, calendar="gregorian")
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=None,
        )
        self.assertEqual(cell, point)

    def test_datetimelike_bounded_cell(self):
        # Check that cell equality works with bounded cells using different datetime objects
        point = cftime.datetime(2010, 1, 1, calendar="gregorian")
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        self.assertEqual(cell, point)

    def test_datetimelike_calenders_cell(self):
        # Check that equality with a cell with a different calendar
        # raises an error. This is not supported
        point = cftime.datetime(2010, 1, 1, calendar="360_day")
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        with self.assertRaisesRegex(TypeError, "different calendars"):
            cell >= point

    def test_PartialDateTime_other(self):
        cell = Cell(datetime.datetime(2010, 3, 2))
        # A few simple cases.
        self.assertEqual(cell, PartialDateTime(month=3))
        self.assertNotEqual(cell, PartialDateTime(month=3, hour=12))
        self.assertNotEqual(cell, PartialDateTime(month=4))


class Test_contains_point(tests.IrisTest):
    """Test that contains_point works for combinations.

    Combinations of datetime, cf.datatime, and PartialDateTime objects.
    """

    def test_datetime_PartialDateTime_point(self):
        point = PartialDateTime(month=6)
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        self.assertFalse(cell.contains_point(point))

    def test_datetime_cftime_standard_point(self):
        point = cftime.datetime(2010, 6, 15)
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        self.assertTrue(cell.contains_point(point))

    def test_datetime_cftime_360day_point(self):
        point = cftime.datetime(2010, 6, 15, calendar="360_day")
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        with self.assertRaisesRegex(TypeError, "different calendars"):
            cell.contains_point(point)

    def test_cftime_standard_PartialDateTime_point(self):
        point = PartialDateTime(month=6)
        cell = Cell(
            cftime.datetime(2010, 1, 1),
            bound=[
                cftime.datetime(2010, 1, 1),
                cftime.datetime(2011, 1, 1),
            ],
        )
        self.assertFalse(cell.contains_point(point))

    def test_cftime_360day_PartialDateTime_point(self):
        point = PartialDateTime(month=6)
        cell = Cell(
            cftime.datetime(2010, 1, 1, calendar="360_day"),
            bound=[
                cftime.datetime(2010, 1, 1, calendar="360_day"),
                cftime.datetime(2011, 1, 1, calendar="360_day"),
            ],
        )
        self.assertFalse(cell.contains_point(point))

    def test_cftime_standard_datetime_point(self):
        point = datetime.datetime(2010, 6, 1)
        cell = Cell(
            cftime.datetime(2010, 1, 1),
            bound=[
                cftime.datetime(2010, 1, 1),
                cftime.datetime(2011, 1, 1),
            ],
        )
        self.assertTrue(cell.contains_point(point))

    def test_cftime_360day_datetime_point(self):
        point = datetime.datetime(2010, 6, 1)
        cell = Cell(
            cftime.datetime(2010, 1, 1, calendar="360_day"),
            bound=[
                cftime.datetime(2010, 1, 1, calendar="360_day"),
                cftime.datetime(2011, 1, 1, calendar="360_day"),
            ],
        )
        with self.assertRaisesRegex(TypeError, "different calendars"):
            cell.contains_point(point)

    def test_cftime_360_day_cftime_360day_point(self):
        point = cftime.datetime(2010, 6, 15, calendar="360_day")
        cell = Cell(
            cftime.datetime(2010, 1, 1, calendar="360_day"),
            bound=[
                cftime.datetime(2010, 1, 1, calendar="360_day"),
                cftime.datetime(2011, 1, 1, calendar="360_day"),
            ],
        )
        self.assertTrue(cell.contains_point(point))


class Test_numpy_comparison(tests.IrisTest):
    """Unit tests to check that the results of comparisons with numpy types can be
    used as truth values.
    """

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
            self.fail("Result of comparison could not be used as a truth value")

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
            self.fail("Result of comparison could not be used as a truth value")


if __name__ == "__main__":
    tests.main()
