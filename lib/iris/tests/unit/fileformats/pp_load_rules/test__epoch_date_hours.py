# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:func:`iris.fileformats.pp_load_rules._epoch_date_hours`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import cf_units
from cf_units import Unit
from cftime import datetime as nc_datetime

from iris.fileformats.pp_load_rules import (
    _epoch_date_hours as epoch_hours_call,
)

#
# Run tests for each of the possible calendars from PPfield.calendar().
# Test year=0 and all=0 cases, plus "normal" dates, for each calendar.
# Result values are the same as from 'date2num' in cftime version <= 1.0.1.
#


class TestEpochHours__gregorian(tests.IrisTest):
    def setUp(self):
        self.hrs_unit = Unit(
            "hours since epoch", calendar=cf_units.CALENDAR_GREGORIAN
        )

    def test_1970_1_1(self):
        test_date = nc_datetime(1970, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, 0.0)

    def test_ymd_1_1_1(self):
        test_date = nc_datetime(1, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17259936.0)

    def test_year_0(self):
        test_date = nc_datetime(0, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17268720.0)

    def test_ymd_0_0_0(self):
        test_date = nc_datetime(0, 0, 0)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17269488.0)

    def test_ymd_0_preserves_timeofday(self):
        hrs, mins, secs, usecs = (7, 13, 24, 335772)
        hours_in_day = (
            hrs + 1.0 / 60 * mins + 1.0 / 3600 * secs + (1.0e-6) / 3600 * usecs
        )
        test_date = nc_datetime(
            0, 0, 0, hour=hrs, minute=mins, second=secs, microsecond=usecs
        )
        result = epoch_hours_call(self.hrs_unit, test_date)
        # NOTE: the calculation is only accurate to approx +/- 0.5 seconds
        # in such a large number of hours -- even 0.1 seconds is too fine.
        absolute_tolerance = 0.5 / 3600
        self.assertArrayAllClose(
            result, -17269488.0 + hours_in_day, rtol=0, atol=absolute_tolerance
        )


class TestEpochHours__360day(tests.IrisTest):
    def setUp(self):
        self.hrs_unit = Unit(
            "hours since epoch", calendar=cf_units.CALENDAR_360_DAY
        )

    def test_1970_1_1(self):
        test_date = nc_datetime(1970, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, 0.0)

    def test_ymd_1_1_1(self):
        test_date = nc_datetime(1, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17012160.0)

    def test_year_0(self):
        test_date = nc_datetime(0, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17020800.0)

    def test_ymd_0_0_0(self):
        test_date = nc_datetime(0, 0, 0)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17021544.0)


class TestEpochHours__365day(tests.IrisTest):
    def setUp(self):
        self.hrs_unit = Unit(
            "hours since epoch", calendar=cf_units.CALENDAR_365_DAY
        )

    def test_1970_1_1(self):
        test_date = nc_datetime(1970, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, 0.0)

    def test_ymd_1_1_1(self):
        test_date = nc_datetime(1, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17248440.0)

    def test_year_0(self):
        test_date = nc_datetime(0, 1, 1)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17257200.0)

    def test_ymd_0_0_0(self):
        test_date = nc_datetime(0, 0, 0)
        result = epoch_hours_call(self.hrs_unit, test_date)
        self.assertEqual(result, -17257968.0)


class TestEpochHours__invalid_calendar(tests.IrisTest):
    def test_bad_calendar(self):
        # Setup a unit with an unrecognised calendar
        hrs_unit = Unit(
            "hours since epoch", calendar=cf_units.CALENDAR_ALL_LEAP
        )
        # Test against a date with year=0, which requires calendar correction.
        test_date = nc_datetime(0, 1, 1)
        # Check that this causes an error.
        with self.assertRaisesRegex(ValueError, "unrecognised calendar"):
            epoch_hours_call(hrs_unit, test_date)


if __name__ == "__main__":
    tests.main()
