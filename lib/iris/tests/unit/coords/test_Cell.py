# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the :class:`iris.coords.Cell` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import datetime

import mock

from iris.coords import Cell


class Test___common_cmp__(tests.IrisTest):
    def test_datetime_ordering(self):
        # Check that cell comparison works with objects with a "timetuple".
        dt = mock.Mock(timetuple=mock.Mock())
        cell = Cell(datetime.datetime(2010, 3, 21))
        with mock.patch('operator.gt') as gt:
            _ = cell > dt
        gt.assert_called_once_with(cell.point, dt)

        # Now check that the existence of timetuple is causing that.
        del dt.timetuple
        with self.assertRaisesRegexp(ValueError,
                                     'Unexpected type of other <(.*)>'):
            _ = cell > dt

    def test_datetime_equality(self):
        # Check that cell equality works with objects with a "timetuple".
        dt = mock.Mock(timetuple=mock.Mock())
        cell = mock.MagicMock(spec=Cell, point=datetime.datetime(2010, 3, 21),
                              bound=None)
        _ = cell == dt
        cell.__eq__.assert_called_once_with(dt)


if __name__ == '__main__':
    tests.main()
