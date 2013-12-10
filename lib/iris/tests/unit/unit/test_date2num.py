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
"""Unit tests for the `iris.unit.date2num` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import datetime
import mock

import iris.unit as unit


class TestAll(tests.IrisTest):
    def test_date_date2num(self):
        # Passing a date to date2num.
        dt = datetime.datetime(1, 1, 1)
        units = 'hours since epoch'
        cal = unit.CALENDAR_STANDARD

        with mock.patch('netcdftime.date2num') as date2num_patch:
            unit.date2num(dt, units, cal)
        self.assertTrue(date2num_patch.called)

    def test_num_date2num(self):
        # Passing a numeric to date2num.
        inp = 1
        units = 'hours since epoch'
        cal = unit.CALENDAR_STANDARD

        with mock.patch('warnings.warn') as warn:
            res = unit.date2num(inp, units, cal)
        msg = 'date2num has not been given a suitable datetime-like object'
        warn.assert_any_call(msg)
        self.assertEqual(res, inp)


if __name__ == "__main__":
    tests.main()
