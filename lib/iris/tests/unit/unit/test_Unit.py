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
"""Unit tests for the `iris.unit.Unit` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import datetime

from iris.unit import Unit


class Test_num2date(tests.IrisTest):
    def setUp(self):
        self.unit = mock.Mock(spec=Unit)

    def test_num_num2date(self):
        # Passing a num to num2date.
        Unit.num2date(self.unit, 1)
        self.assertTrue(self.unit.utime.called)

    def test_date_num2date(self):
        date = datetime.datetime(1, 1, 1)
        # Passing a date to num2date.
        with mock.patch('warnings.warn') as warn:
            res = Unit('hours since epoch').num2date(date)
        msg = 'num2date has not been given a suitable numeric value'
        warn.assert_any_call(msg)
        self.assertEqual(res, date)


class Test_date2num(tests.IrisTest):
    def setUp(self):
        self.unit = mock.Mock(spec=Unit)

    def test_date_date2num(self):
        # Passing a date to date2num.
        Unit.date2num(self.unit, datetime.datetime(1, 1, 1))
        self.assertTrue(self.unit.utime.called)

    def test_num_date2num(self):
        # Passing a numeric to date2num.
        with mock.patch('warnings.warn') as warn:
            res = Unit('hours since epoch').date2num(1)
        msg = 'date2num has not been given a suitable datetime-like object'
        warn.assert_any_call(msg)
        self.assertEqual(res, 1)


if __name__ == "__main__":
    tests.main()
