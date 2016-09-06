# (C) British Crown Copyright 2016, Met Office
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
"""Test function :func:`iris.util._num2date_to_nearest_second`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import datetime

from cf_units import Unit
import numpy as np
import netcdftime

from iris.util import _num2date_to_nearest_second


class Test(tests.IrisTest):
    def setup_units(self, calendar):
        self.useconds = Unit('seconds since epoch', calendar)
        self.uminutes = Unit('minutes since epoch', calendar)
        self.uhours = Unit('hours since epoch', calendar)
        self.udays = Unit('days since epoch', calendar)

    def check_dates(self, nums, units, expected):
        for num, unit, exp in zip(nums, units, expected):
            res = _num2date_to_nearest_second(num, unit)
            self.assertEqual(exp, res)

    # Gregorian Calendar tests

    def test_simple_gregorian(self):
        self.setup_units('gregorian')
        nums = [20., 40.,
                75., 150.,
                8., 16.,
                300., 600.]
        units = [self.useconds, self.useconds,
                 self.uminutes, self.uminutes,
                 self.uhours, self.uhours,
                 self.udays, self.udays]
        expected = [datetime.datetime(1970, 1, 1, 0, 0, 20),
                    datetime.datetime(1970, 1, 1, 0, 0, 40),
                    datetime.datetime(1970, 1, 1, 1, 15),
                    datetime.datetime(1970, 1, 1, 2, 30),
                    datetime.datetime(1970, 1, 1, 8),
                    datetime.datetime(1970, 1, 1, 16),
                    datetime.datetime(1970, 10, 28),
                    datetime.datetime(1971, 8, 24)]

        self.check_dates(nums, units, expected)

    def test_fractional_gregorian(self):
        self.setup_units('gregorian')
        nums = [5./60., 10./60.,
                15./60., 30./60.,
                8./24., 16./24.]
        units = [self.uminutes, self.uminutes,
                 self.uhours, self.uhours,
                 self.udays, self.udays]
        expected = [datetime.datetime(1970, 1, 1, 0, 0, 5),
                    datetime.datetime(1970, 1, 1, 0, 0, 10),
                    datetime.datetime(1970, 1, 1, 0, 15),
                    datetime.datetime(1970, 1, 1, 0, 30),
                    datetime.datetime(1970, 1, 1, 8),
                    datetime.datetime(1970, 1, 1, 16)]

        self.check_dates(nums, units, expected)

    def test_fractional_second_gregorian(self):
        self.setup_units('gregorian')
        nums = [0.25, 0.5, 0.75,
                1.5, 2.5, 3.5, 4.5]
        units = [self.useconds]*7
        expected = [datetime.datetime(1970, 1, 1, 0, 0, 0),
                    datetime.datetime(1970, 1, 1, 0, 0, 1),
                    datetime.datetime(1970, 1, 1, 0, 0, 1),
                    datetime.datetime(1970, 1, 1, 0, 0, 2),
                    datetime.datetime(1970, 1, 1, 0, 0, 3),
                    datetime.datetime(1970, 1, 1, 0, 0, 4),
                    datetime.datetime(1970, 1, 1, 0, 0, 5)]

        self.check_dates(nums, units, expected)

    # 360 day Calendar tests

    def test_simple_360_day(self):
        self.setup_units('360_day')
        nums = [20., 40.,
                75., 150.,
                8., 16.,
                300., 600.]
        units = [self.useconds, self.useconds,
                 self.uminutes, self.uminutes,
                 self.uhours, self.uhours,
                 self.udays, self.udays]
        expected = [netcdftime.datetime(1970, 1, 1, 0, 0, 20),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 40),
                    netcdftime.datetime(1970, 1, 1, 1, 15),
                    netcdftime.datetime(1970, 1, 1, 2, 30),
                    netcdftime.datetime(1970, 1, 1, 8),
                    netcdftime.datetime(1970, 1, 1, 16),
                    netcdftime.datetime(1970, 11, 1),
                    netcdftime.datetime(1971, 9, 1)]

        self.check_dates(nums, units, expected)

    def test_fractional_360_day(self):
        self.setup_units('360_day')
        nums = [5./60., 10./60.,
                15./60., 30./60.,
                8./24., 16./24.]
        units = [self.uminutes, self.uminutes,
                 self.uhours, self.uhours,
                 self.udays, self.udays]
        expected = [netcdftime.datetime(1970, 1, 1, 0, 0, 5),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 10),
                    netcdftime.datetime(1970, 1, 1, 0, 15),
                    netcdftime.datetime(1970, 1, 1, 0, 30),
                    netcdftime.datetime(1970, 1, 1, 8),
                    netcdftime.datetime(1970, 1, 1, 16)]

        self.check_dates(nums, units, expected)

    def test_fractional_second_360_day(self):
        self.setup_units('360_day')
        nums = [0.25, 0.5, 0.75,
                1.5, 2.5, 3.5, 4.5]
        units = [self.useconds]*7
        expected = [netcdftime.datetime(1970, 1, 1, 0, 0, 0),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 1),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 1),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 2),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 3),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 4),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 5)]

        self.check_dates(nums, units, expected)

    # 365 day Calendar tests

    def test_simple_365_day(self):
        self.setup_units('365_day')
        nums = [20., 40.,
                75., 150.,
                8., 16.,
                300., 600.]
        units = [self.useconds, self.useconds,
                 self.uminutes, self.uminutes,
                 self.uhours, self.uhours,
                 self.udays, self.udays]
        expected = [netcdftime.datetime(1970, 1, 1, 0, 0, 20),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 40),
                    netcdftime.datetime(1970, 1, 1, 1, 15),
                    netcdftime.datetime(1970, 1, 1, 2, 30),
                    netcdftime.datetime(1970, 1, 1, 8),
                    netcdftime.datetime(1970, 1, 1, 16),
                    netcdftime.datetime(1970, 10, 28),
                    netcdftime.datetime(1971, 8, 24)]

        self.check_dates(nums, units, expected)

    def test_fractional_365_day(self):
        self.setup_units('365_day')
        nums = [5./60., 10./60.,
                15./60., 30./60.,
                8./24., 16./24.]
        units = [self.uminutes, self.uminutes,
                 self.uhours, self.uhours,
                 self.udays, self.udays]

        expected = [netcdftime.datetime(1970, 1, 1, 0, 0, 5),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 10),
                    netcdftime.datetime(1970, 1, 1, 0, 15),
                    netcdftime.datetime(1970, 1, 1, 0, 30),
                    netcdftime.datetime(1970, 1, 1, 8),
                    netcdftime.datetime(1970, 1, 1, 16)]

        self.check_dates(nums, units, expected)

    def test_fractional_second_365_day(self):
        self.setup_units('365_day')
        nums = [0.25, 0.5, 0.75,
                1.5, 2.5, 3.5, 4.5]
        units = [self.useconds]*7
        expected = [netcdftime.datetime(1970, 1, 1, 0, 0, 0),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 1),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 1),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 2),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 3),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 4),
                    netcdftime.datetime(1970, 1, 1, 0, 0, 5)]

        self.check_dates(nums, units, expected)

if __name__ == '__main__':
    tests.main()
