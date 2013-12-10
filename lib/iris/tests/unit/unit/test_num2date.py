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
"""Unit tests for the `iris.unit.num2date` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import datetime
import mock

import iris.unit as unit


class TestAll(tests.IrisTest):
    def test_num_num2date(self):
        # Passing a numeric to num2date.
        inp = 1
        units = 'hours since epoch'
        cal = unit.CALENDAR_STANDARD

        with mock.patch('netcdftime.num2date') as num2date_patch:
            unit.num2date(inp, units, cal)
        self.assertTrue(num2date_patch.called)

    def test_date_num2date(self):
        # Passing a date to num2date.
        inp = datetime.datetime(1, 1, 1)
        units = 'hours since epoch'
        cal = unit.CALENDAR_STANDARD

        with mock.patch('warnings.warn') as warn:
            res = unit.num2date(inp, units, cal)
        msg = 'num2date has not been given a suitable numeric value'
        warn.assert_any_call(msg)
        self.assertEqual(res, inp)


if __name__ == "__main__":
    tests.main()
