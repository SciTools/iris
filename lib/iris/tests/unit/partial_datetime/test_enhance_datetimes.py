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
"""Unit tests for the :func:`iris.partial_datetime.enhance_datetimes`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris.partial_datetime as ipd


class TestAll(tests.IrisTest):
    def setUp(self):
        _DatetimeWrap_patch = mock.patch('iris.partial_datetime.DatetimeWrap')
        _DatetimeWrap_patch.start()
        self.addCleanup(_DatetimeWrap_patch.stop)

        self.unit = mock.Mock()
        self.datetime = mock.Mock()

    def test_make_iterable(self):
        # When supplied with a single datetime, ensure that an iterable is
        # returned.
        res = ipd.enhance_datetimes(self.unit, self.datetime)
        self.assertTrue(hasattr(res, '__iter__'))
        ipd.DatetimeWrap.assert_called_once_with(self.datetime, self.unit)

    def test_already_iterable(self):
        # When supplied with an iterable of datetime-like objects, ensure that
        # it DatetimeWrap is called on each of them.
        res = ipd.enhance_datetimes(self.unit, [self.datetime])
        self.assertTrue(hasattr(res, '__iter__'))
        ipd.DatetimeWrap.assert_called_once_with(self.datetime, self.unit)


if __name__ == '__main__':
    tests.main()
