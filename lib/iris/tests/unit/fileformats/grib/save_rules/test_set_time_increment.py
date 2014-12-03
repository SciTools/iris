# (C) British Crown Copyright 2014, Met Office
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
"""
Unit tests for
:func:`iris.fileformats.grib._save_rules.set_time_increment`

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock

from iris.coords import CellMethod
from iris.fileformats.grib._save_rules import set_time_increment


class Test(tests.IrisTest):
    @mock.patch.object(gribapi, 'grib_set')
    def test_no_intervals(self, mock_set):
        cell_method = CellMethod('sum', 'time')
        set_time_increment(cell_method, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 'indicatorOfUnitForTimeIncrement', 255)
        mock_set.assert_any_call(mock.sentinel.grib, 'timeIncrement', 0)

    @mock.patch.object(gribapi, 'grib_set')
    def test_area(self, mock_set):
        cell_method = CellMethod('sum', 'area', '25 km')
        set_time_increment(cell_method, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 'indicatorOfUnitForTimeIncrement', 255)
        mock_set.assert_any_call(mock.sentinel.grib, 'timeIncrement', 0)

    @mock.patch.object(gribapi, 'grib_set')
    def test_multiple_intervals(self, mock_set):
        cell_method = CellMethod('sum', 'time', ('1 hour', '24 hour'))
        set_time_increment(cell_method, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 'indicatorOfUnitForTimeIncrement', 255)
        mock_set.assert_any_call(mock.sentinel.grib, 'timeIncrement', 0)

    @mock.patch.object(gribapi, 'grib_set')
    def test_hr(self, mock_set):
        cell_method = CellMethod('sum', 'time', '23 hr')
        set_time_increment(cell_method, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 'indicatorOfUnitForTimeIncrement', 1)
        mock_set.assert_any_call(mock.sentinel.grib, 'timeIncrement', 23)

    @mock.patch.object(gribapi, 'grib_set')
    def test_hour(self, mock_set):
        cell_method = CellMethod('sum', 'time', '24 hour')
        set_time_increment(cell_method, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 'indicatorOfUnitForTimeIncrement', 1)
        mock_set.assert_any_call(mock.sentinel.grib, 'timeIncrement', 24)

    @mock.patch.object(gribapi, 'grib_set')
    def test_hours(self, mock_set):
        cell_method = CellMethod('sum', 'time', '25 hours')
        set_time_increment(cell_method, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 'indicatorOfUnitForTimeIncrement', 1)
        mock_set.assert_any_call(mock.sentinel.grib, 'timeIncrement', 25)

    @mock.patch.object(gribapi, 'grib_set')
    def test_fractional_hours(self, mock_set):
        cell_method = CellMethod('sum', 'time', '25.9 hours')
        with mock.patch('warnings.warn') as warn:
            set_time_increment(cell_method, mock.sentinel.grib)
        warn.assert_called_once_with('Truncating floating point timeIncrement '
                                     '25.9 to integer value 25')
        mock_set.assert_any_call(mock.sentinel.grib,
                                 'indicatorOfUnitForTimeIncrement', 1)
        mock_set.assert_any_call(mock.sentinel.grib, 'timeIncrement', 25)


if __name__ == "__main__":
    tests.main()
