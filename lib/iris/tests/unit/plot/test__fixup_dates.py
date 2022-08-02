# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot._fixup_dates` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import datetime

from cf_units import Unit
import cftime

from iris.coords import AuxCoord
from iris.plot import _fixup_dates


class Test(tests.IrisTest):
    def test_standard_calendar(self):
        unit = Unit("hours since 2000-04-13 00:00:00", calendar="standard")
        coord = AuxCoord([1, 3, 6], "time", units=unit)
        result = _fixup_dates(coord, coord.points)
        self.assertIsInstance(result[0], datetime.datetime)
        expected = [
            datetime.datetime(2000, 4, 13, 1),
            datetime.datetime(2000, 4, 13, 3),
            datetime.datetime(2000, 4, 13, 6),
        ]
        self.assertArrayEqual(result, expected)

    def test_standard_calendar_sub_second(self):
        unit = Unit("seconds since 2000-04-13 00:00:00", calendar="standard")
        coord = AuxCoord([1, 1.25, 1.5], "time", units=unit)
        result = _fixup_dates(coord, coord.points)
        self.assertIsInstance(result[0], datetime.datetime)
        expected = [
            datetime.datetime(2000, 4, 13, 0, 0, 1),
            datetime.datetime(2000, 4, 13, 0, 0, 1),
            datetime.datetime(2000, 4, 13, 0, 0, 2),
        ]
        self.assertArrayEqual(result, expected)

    @tests.skip_nc_time_axis
    def test_360_day_calendar(self):
        calendar = "360_day"
        unit = Unit("days since 2000-02-25 00:00:00", calendar=calendar)
        coord = AuxCoord([3, 4, 5], "time", units=unit)
        result = _fixup_dates(coord, coord.points)
        expected_datetimes = [
            cftime.datetime(2000, 2, 28, calendar=calendar),
            cftime.datetime(2000, 2, 29, calendar=calendar),
            cftime.datetime(2000, 2, 30, calendar=calendar),
        ]
        self.assertArrayEqual(result, expected_datetimes)

    @tests.skip_nc_time_axis
    def test_365_day_calendar(self):
        calendar = "365_day"
        unit = Unit("minutes since 2000-02-25 00:00:00", calendar=calendar)
        coord = AuxCoord([30, 60, 150], "time", units=unit)
        result = _fixup_dates(coord, coord.points)
        expected_datetimes = [
            cftime.datetime(2000, 2, 25, 0, 30, calendar=calendar),
            cftime.datetime(2000, 2, 25, 1, 0, calendar=calendar),
            cftime.datetime(2000, 2, 25, 2, 30, calendar=calendar),
        ]
        self.assertArrayEqual(result, expected_datetimes)

    @tests.skip_nc_time_axis
    def test_360_day_calendar_attribute(self):
        calendar = "360_day"
        unit = Unit("days since 2000-02-01 00:00:00", calendar=calendar)
        coord = AuxCoord([0, 3, 6], "time", units=unit)
        result = _fixup_dates(coord, coord.points)
        self.assertEqual(result[0].calendar, calendar)


if __name__ == "__main__":
    tests.main()
