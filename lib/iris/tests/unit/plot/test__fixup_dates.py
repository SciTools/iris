# (C) British Crown Copyright 2016 - 2018, Met Office
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
"""Unit tests for the `iris.plot._fixup_dates` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from cf_units import Unit
import datetime
import cftime

from iris.coords import AuxCoord
from iris.plot import _fixup_dates


class Test(tests.IrisTest):
    def coord(self, calendar):
        unit = Unit('hours since 2000-04-13 00:00:00', calendar=calendar)
        coord = AuxCoord([1, 2, 3], 'time', units=unit)

    def test_gregorian_calendar(self):
        unit = Unit('hours since 2000-04-13 00:00:00', calendar='gregorian')
        coord = AuxCoord([1, 3, 6], 'time', units=unit)
        result = _fixup_dates(coord, coord.points)
        expected = [datetime.datetime(2000, 4, 13, 1),
                    datetime.datetime(2000, 4, 13, 3),
                    datetime.datetime(2000, 4, 13, 6)]
        self.assertArrayEqual(result, expected)

    def test_gregorian_calendar_sub_second(self):
        unit = Unit('seconds since 2000-04-13 00:00:00', calendar='gregorian')
        coord = AuxCoord([1, 1.25, 1.5], 'time', units=unit)
        result = _fixup_dates(coord, coord.points)
        expected = [datetime.datetime(2000, 4, 13, 0, 0, 1),
                    datetime.datetime(2000, 4, 13, 0, 0, 1),
                    datetime.datetime(2000, 4, 13, 0, 0, 2)]
        self.assertArrayEqual(result, expected)

    @tests.skip_nc_time_axis
    def test_360_day_calendar(self):
        unit = Unit('days since 2000-02-25 00:00:00', calendar='360_day')
        coord = AuxCoord([3, 4, 5], 'time', units=unit)
        result = _fixup_dates(coord, coord.points)
        expected_datetimes = [cftime.datetime(2000, 2, 28),
                              cftime.datetime(2000, 2, 29),
                              cftime.datetime(2000, 2, 30)]
        self.assertArrayEqual([cdt.datetime for cdt in result],
                              expected_datetimes)

    @tests.skip_nc_time_axis
    def test_365_day_calendar(self):
        unit = Unit('minutes since 2000-02-25 00:00:00', calendar='365_day')
        coord = AuxCoord([30, 60, 150], 'time', units=unit)
        result = _fixup_dates(coord, coord.points)
        expected_datetimes = [cftime.datetime(2000, 2, 25, 0, 30),
                              cftime.datetime(2000, 2, 25, 1, 0),
                              cftime.datetime(2000, 2, 25, 2, 30)]
        self.assertArrayEqual([cdt.datetime for cdt in result],
                              expected_datetimes)

    @tests.skip_nc_time_axis
    def test_360_day_calendar_attribute(self):
        unit = Unit('days since 2000-02-01 00:00:00', calendar='360_day')
        coord = AuxCoord([0, 3, 6], 'time', units=unit)
        result = _fixup_dates(coord, coord.points)
        self.assertEqual(result[0].calendar, '360_day')


if __name__ == "__main__":
    tests.main()
