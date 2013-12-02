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
"""Unit tests for the :func:`iris.coords._datetime_numeric_comparison`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from datetime import datetime
import mock
import netcdftime

import iris.coords as icoords
from iris.unit import Unit


class TestDatetime(tests.IrisTest):
    def setUp(self):
        self.unit = Unit('hours since epoch')
        self.datetimes = [datetime(1, 1, 1, 1), datetime(1, 1, 1, 2)]
        self.res = icoords._datetime_numeric_comparison(
            self.unit, self.datetimes)

    def test_returning_subclass(self):
        # This test ensures that we return a subclass of datetime object which
        # has methods suitable of returning a numeric representation, holding
        # unit information and issuing deprecation warnings.
        for item in self.res:
            self.assertTrue(isinstance(item, icoords._Idatetime))
            self.assertTrue(hasattr(item, '__float__'))
            self.assertTrue(hasattr(item, 'unit'))

    def test_float_return_value(self):
        for orig, ret in zip(self.datetimes, self.res):
            self.assertEqual(self.unit.date2num(orig), float(ret))

    def test_equality(self):
        for orig, ret in zip(self.datetimes, self.res):
            self.assertEqual(orig, ret)

    def test_equality_numeric(self):
        # Equality of datetime subclass with a numeric (deprecation warning).
        with mock.patch('warnings.warn') as warn:
            self.assertFalse(self.res[0] == 1)
        msg = ('Comparing datetime objects with numeric objects (int, '
               'float) is being deprecated, consider switching to using '
               'iris.pdatetime.PartialDateTime objects')
        warn.assert_called_with(msg)


class TestNetcdftimeDatetime(tests.IrisTest):
    def setUp(self):
        self.unit = Unit('hours since epoch')
        self.datetimes = [netcdftime.datetime(1, 1, 1, 1),
                          netcdftime.datetime(1, 1, 1, 2)]
        self.res = icoords._datetime_numeric_comparison(
            self.unit, self.datetimes)

    def test_returning_added_methods(self):
        # This test ensures that we return the netcdf datetime with additional
        # methods for returning a numeric representation, holding
        # unit information and issuing deprecation warnings.
        for item in self.res:
            self.assertTrue(isinstance(item, netcdftime.datetime))
            self.assertTrue(hasattr(item, '__float__'))
            self.assertTrue(hasattr(item, 'unit'))

    def test_float_return_value(self):
        for orig, ret in zip(self.datetimes, self.res):
            self.assertEqual(self.unit.date2num(orig), float(ret))

    def test_equality(self):
        for orig, ret in zip(self.datetimes, self.res):
            self.assertEqual(orig, ret)

    def test_equality_numeric(self):
        # Equality of modified netcdftime.datetime object with a numeric
        # (deprecation warning).
        with mock.patch('warnings.warn') as warn:
            self.assertFalse(self.res[0] == 1)
        msg = ('Comparing netcdftime.datetime objects with numeric objects '
               '(int, float) is being deprecated, consider switching to using '
               'iris.pdatetime.PartialDateTime objects')
        warn.assert_called_with(msg)


if __name__ == '__main__':
    tests.main()
