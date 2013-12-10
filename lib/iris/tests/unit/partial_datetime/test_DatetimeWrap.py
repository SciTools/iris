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
"""Unit tests for the :func:`iris.partial_datetime.DatetimeWrap` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from datetime import datetime
import mock
import operator

import iris.partial_datetime as ipd


class Test___getattribute__(tests.IrisTest):
    def setUp(self):
        self.datetime = mock.Mock(spec=datetime, a=1, b=2)
        self.unit = mock.Mock()
        self.wrap = ipd.DatetimeWrap(self.datetime, self.unit)

    def test_access_wrapper_class_attributes(self):
        # Ensure that we can extract the attributes of the wrapper class.
        self.assertEqual(self.wrap.unit, self.unit)

    def test_access_wrapped_class_attributes(self):
        # Ensure that we can extract the attributes of the wrapped class.
        self.assertEqual(self.wrap.a, 1)


class Test__compare(tests.IrisTest):
    def setUp(self):
        self.datetime = mock.Mock(spec=datetime, a=1, b=2)
        self.unit = mock.Mock()
        self.unit.date2num = mock.Mock(return_value=5.0)
        self.wrap = ipd.DatetimeWrap(self.datetime, self.unit)
        self.operator = mock.Mock(name='operator')

    def test_warning_issued(self):
        msg = ('Comparing datetime objects with numeric objects (int, '
               'float) is being deprecated, consider switching to using '
               'iris.partial_datetime.PartialDateTime objects')
        with mock.patch('warnings.warn') as warn:
            self.wrap._compare(self.operator, 1)

        self.assertIn(mock.call(msg, DeprecationWarning), warn.call_args_list)
        self.operator.assert_called_once_with(
            self.unit.date2num.return_value, 1)

    def test___gt__(self):
        ret = self.wrap._compare(operator.gt, 1)
        self.assertTrue(ret)

    def test___lt__(self):
        ret = self.wrap._compare(operator.lt, 1)
        self.assertFalse(ret)

    def test___eq__(self):
        ret = self.wrap._compare(operator.eq, 1)
        self.assertFalse(ret)

    def test___eq__v2(self):
        ret = self.wrap._compare(operator.eq, 5)
        self.assertTrue(ret)


if __name__ == '__main__':
    tests.main()
