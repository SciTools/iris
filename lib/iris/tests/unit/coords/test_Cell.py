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
"""Unit tests for the :class:`iris.coords.Cell`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import operator

from datetime import datetime
import netcdftime
import numpy as np

from iris.coords import Cell
from iris.pdatetime import PartialDateTime
import iris.tests.unit as unit


class Test___common_cmp__(tests.IrisTest):
    def setUp(self):
        dt = mock.Mock(spec=datetime)
        self.cell = Cell(point=dt, bound=[dt, dt])

    def test_incompatible_comparison(self):
        # Ensure that a TypeError is raised when inappropriate type is for
        # comparison.
        dt = mock.Mock(spec=datetime)
        test_cell = Cell(point=dt, bound=[dt, dt])

        op = mock.Mock(name='operator')
        other = mock.Mock(name='comparison_object')
        with mock.patch('warnings.warn') as warn:
            with self.assertRaises(ValueError) as err:
                test_cell.__common_cmp__(other, op)
        self.assertEqual(err.exception.message, "Unexpected type of other")
        self.assertFalse(warn.called)

    def test_old_comparison_warning(self):
        # Ensure that warning is issued when comparing a datetime cell point
        # with a non datetime compatible object.
        dt = mock.Mock(spec=datetime)
        test_cell = Cell(point=dt, bound=[dt, dt])

        op = operator.gt
        other = 1
        with mock.patch('warnings.warn') as warn:
            test_cell.__common_cmp__(other, op)
        msg = ('A comparison is taking place between a cell with datetimes '
               'and a numeric. Is this an old style constraint? You know '
               'datetimes are much richer?')
        warn.assert_called_with(msg)

    def test_comparison_datetime_with_datetime(self):
        # Ensure that no warning is issued when comparing datetime compatible
        # object comparison is made.
        dt = mock.Mock(spec=datetime)
        test_cell = Cell(point=dt, bound=[dt, dt])

        op = operator.gt
        other = mock.Mock(spec=datetime)
        with mock.patch('warnings.warn') as warn:
            test_cell.__common_cmp__(other, op)
        self.assertFalse(warn.called)


class Test___eq__(tests.IrisTest):
    def test_datetime_cell(self):
        dt = mock.Mock(spec=datetime)
        cell = Cell(point='dummy', bound=None)

        with unit.patched_isinstance(return_value=True) as new_isinstance:
            cell.__eq__(dt)
        new_isinstance.assert_called_once_with(
            dt, (int, float, np.number, PartialDateTime,
                 PartialDateTime.known_time_implementations))

    def test_PartialDateTime_cell(self):
        dt = mock.Mock(spec=PartialDateTime)
        cell = Cell(point='dummy', bound=None)

        with unit.patched_isinstance(return_value=True) as new_isinstance:
            cell.__eq__(dt)
        new_isinstance.assert_called_once_with(
            dt, (int, float, np.number, PartialDateTime,
                 PartialDateTime.known_time_implementations))


if __name__ == '__main__':
    tests.main()
