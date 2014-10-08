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
Tests for function
:func:`iris.fileformats.grib._load_convert.statistical_forecast_period`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import datetime
import mock

from iris.fileformats.grib._load_convert import \
    statistical_forecast_period_coord


class Test(tests.IrisTest):
    def setUp(self):
        module = 'iris.fileformats.grib._load_convert'
        self.module = module
        self.patch_hindcast = self.patch(module + '._hindcast_fix')
        self.forecast_seconds = 0.0
        self.forecast_units = mock.Mock()
        self.forecast_units.convert = lambda x, y: self.forecast_seconds
        self.patch(module + '.time_range_unit',
                   return_value=self.forecast_units)
        self.frt_coord = mock.Mock()
        self.frt_coord.points = [1]
        self.frt_coord.units.num2date = mock.Mock(
            return_value=datetime.datetime(2010, 2, 3))
        self.section = {}
        self.section['yearOfEndOfOverallTimeInterval'] = 2010
        self.section['monthOfEndOfOverallTimeInterval'] = 2
        self.section['dayOfEndOfOverallTimeInterval'] = 3
        self.section['hourOfEndOfOverallTimeInterval'] = 8
        self.section['minuteOfEndOfOverallTimeInterval'] = 0
        self.section['secondOfEndOfOverallTimeInterval'] = 0
        self.section['forecastTime'] = mock.Mock()
        self.section['indicatorOfUnitOfTimeRange'] = mock.Mock()

    def test_basic(self):
        coord = statistical_forecast_period_coord(self.section,
                                                  self.frt_coord)
        self.assertEqual(coord.standard_name, 'forecast_period')
        self.assertEqual(coord.units, 'hours')
        self.assertArrayAlmostEqual(coord.points, [4.0])
        self.assertArrayAlmostEqual(coord.bounds, [[0.0, 8.0]])

    def test_with_hindcast(self):
        coord = statistical_forecast_period_coord(self.section,
                                                  self.frt_coord)
        self.assertEqual(self.patch_hindcast.call_count, 1)

    def test_no_hindcast(self):
        self.patch(self.module + '.options.support_hindcast_values', False)
        coord = statistical_forecast_period_coord(self.section,
                                                  self.frt_coord)
        self.assertEqual(self.patch_hindcast.call_count, 0)


if __name__ == '__main__':
    tests.main()
