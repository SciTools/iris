# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coords.Cell` class."""

import datetime

import cftime
import numpy as np
import pytest

from iris.coords import Cell
from iris.time import PartialDateTime


class Test___common_cmp__:
    def assert_raises_on_comparison(self, cell, other, exception_type, regexp):
        with pytest.raises(exception_type, match=regexp):
            cell < other
        with pytest.raises(exception_type, match=regexp):
            cell <= other
        with pytest.raises(exception_type, match=regexp):
            cell > other
        with pytest.raises(exception_type, match=regexp):
            cell >= other

    def test_partial_date_time_bounded_cell(self):
        # Check bounded cell comparisons to a PartialDateTime
        dt = PartialDateTime(month=6)
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        assert dt > cell
        assert dt >= cell
        assert cell < dt
        assert cell <= dt

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

    def test_partial_date_time_unbounded_cell(self):
        # Check that cell comparison works with PartialDateTimes.
        dt = PartialDateTime(month=6)
        cell = Cell(cftime.datetime(2010, 3, 1))
        assert cell < dt
        assert dt > cell
        assert cell <= dt
        assert dt >= cell

    def test_datetime_unbounded_cell(self):
        # Check that cell comparison works with datetimes & cftimes.
        dt = datetime.datetime(2000, 6, 15)
        cell = Cell(cftime.datetime(2000, 1, 1))
        assert dt > cell
        assert dt >= cell
        assert cell < dt
        assert cell <= dt

    def test_0_d_numpy_array(self):
        # Check that cell comparison works with 0D numpy arrays

        cell = Cell(1.3)

        assert np.array(1.5) > cell
        assert np.array(1.1) < cell
        assert np.array(1.3) >= cell
        assert np.array(1.3) <= cell

    def test_len_1_numpy_array(self):
        # Check that cell comparison works with numpy arrays of len=1

        cell = Cell(1.3)

        assert np.array([1.5]) > cell
        assert np.array([1.1]) < cell
        assert np.array([1.3]) >= cell
        assert np.array([1.3]) <= cell


class Test___eq__:
    def test_datetimelike(self):
        # Check that cell equality works with different datetime objects
        # using the same calendar
        point = cftime.datetime(2010, 1, 1, calendar="gregorian")
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=None,
        )
        assert cell == point

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
        assert cell == point

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
        with pytest.raises(TypeError, match="different calendars"):
            cell >= point

    def test_partial_date_time_other(self):
        cell = Cell(datetime.datetime(2010, 3, 2))
        # A few simple cases.
        assert cell == PartialDateTime(month=3)
        assert cell != PartialDateTime(month=3, hour=12)
        assert cell != PartialDateTime(month=4)


class Test_contains_point:
    """Test that contains_point works for combinations.

    Combinations of datetime, cf.datatime, and PartialDateTime objects.
    """

    def test_datetime_partial_date_time_point(self):
        point = PartialDateTime(month=6)
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        assert not cell.contains_point(point)

    def test_datetime_cftime_standard_point(self):
        point = cftime.datetime(2010, 6, 15)
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        assert cell.contains_point(point)

    def test_datetime_cftime_360day_point(self):
        point = cftime.datetime(2010, 6, 15, calendar="360_day")
        cell = Cell(
            datetime.datetime(2010, 1, 1),
            bound=[
                datetime.datetime(2010, 1, 1),
                datetime.datetime(2011, 1, 1),
            ],
        )
        with pytest.raises(TypeError, match="different calendars"):
            cell.contains_point(point)

    def test_cftime_standard_partial_date_time_point(self):
        point = PartialDateTime(month=6)
        cell = Cell(
            cftime.datetime(2010, 1, 1),
            bound=[
                cftime.datetime(2010, 1, 1),
                cftime.datetime(2011, 1, 1),
            ],
        )
        assert not cell.contains_point(point)

    def test_cftime_360day_partial_date_time_point(self):
        point = PartialDateTime(month=6)
        cell = Cell(
            cftime.datetime(2010, 1, 1, calendar="360_day"),
            bound=[
                cftime.datetime(2010, 1, 1, calendar="360_day"),
                cftime.datetime(2011, 1, 1, calendar="360_day"),
            ],
        )
        assert not cell.contains_point(point)

    def test_cftime_standard_datetime_point(self):
        point = datetime.datetime(2010, 6, 1)
        cell = Cell(
            cftime.datetime(2010, 1, 1),
            bound=[
                cftime.datetime(2010, 1, 1),
                cftime.datetime(2011, 1, 1),
            ],
        )
        assert cell.contains_point(point)

    def test_cftime_360day_datetime_point(self):
        point = datetime.datetime(2010, 6, 1)
        cell = Cell(
            cftime.datetime(2010, 1, 1, calendar="360_day"),
            bound=[
                cftime.datetime(2010, 1, 1, calendar="360_day"),
                cftime.datetime(2011, 1, 1, calendar="360_day"),
            ],
        )
        with pytest.raises(TypeError, match="different calendars"):
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
        assert cell.contains_point(point)


class Test_numpy_comparison:
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
            pytest.fail("Result of comparison could not be used as a truth value")

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
            pytest.fail("Result of comparison could not be used as a truth value")


class Test_hashing:
    @pytest.mark.parametrize(
        "point",
        (
            pytest.param(np.float32(1.0), id="float32"),
            pytest.param(np.float64(1.0), id="float64"),
            pytest.param(np.int16(1), id="int16"),
            pytest.param(np.int32(1), id="int32"),
            pytest.param(np.int64(1), id="int64"),
            pytest.param(np.uint16(1), id="uint16"),
            pytest.param(np.uint32(1), id="uint32"),
            pytest.param(np.uint64(1), id="uint64"),
            pytest.param(True, id="bool"),
            pytest.param(np.ma.masked, id="masked"),
            pytest.param(datetime.datetime(2001, 1, 1), id="datetime"),
        ),
    )
    def test_cell_is_hashable(self, point):
        """Test a Cell object is hashable with various point/bound types."""
        # test with no bounds:
        cell = Cell(point=point, bound=None)
        hash(cell)

        # if a numerical type, then test with bounds based on point:
        if isinstance(point, np.number):
            cell = Cell(point=input, bound=((point - 1, point + 1)))
            hash(cell)
