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
"""Test function :func:`iris.util._fix_netcdftime_datetime`."""


# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import netcdftime

import iris.unit
from iris.util import _fix_netcdftime_datetime


class Test(tests.IrisTest):
    # The function happens to work with datetime objects too,
    # so might as well test those in here too.

    def test_greg(self):
        dt = netcdftime.datetime(2001, 12, 31)
        unit = iris.unit.Unit('days since 2000-01-01',
                              calendar=iris.unit.CALENDAR_GREGORIAN)
        day_of_year = _fix_netcdftime_datetime(dt, unit).timetuple().tm_yday
        self.assertEqual(day_of_year, 365)

    def test_greg_leap(self):
        dt = netcdftime.datetime(2000, 12, 31)
        unit = iris.unit.Unit('days since 2000-01-01',
                              calendar=iris.unit.CALENDAR_GREGORIAN)
        day_of_year = _fix_netcdftime_datetime(dt, unit).timetuple().tm_yday
        self.assertEqual(day_of_year, 366)

    def test_360(self):
        dt = netcdftime.datetime(2000, 12, 30)
        unit = iris.unit.Unit('days since 2000-01-01',
                              calendar=iris.unit.CALENDAR_360_DAY)
        self.assertEqual(_fix_netcdftime_datetime(dt, unit).dayofyr, 360)


if __name__ == '__main__':
    tests.main()
