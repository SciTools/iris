# (C) British Crown Copyright 2013 - 2015, Met Office
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
Unit tests for :func:`iris.fileformats.name_loaders.__calc_integration_period`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import datetime

from iris.fileformats.name_loaders import _calc_integration_period


class Tests(tests.IrisTest):
    def test_30_min_av(self):
        time_avgs = ['             30min average']
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (30*60))]
        self.assertEqual(result, expected)

    def test_3_hour_av(self):
        time_avgs = ['          3hr 0min average']
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (3*60*60))]
        self.assertEqual(result, expected)

    def test_3_hour_int(self):
        time_avgs = ['         3hr 0min integral']
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (30*60*60))]
        self.assertEqual(result, expected)

    def test_12_hour_av(self):
        time_avgs = ['         12hr 0min average']
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (12*60*60))]
        self.assertEqual(result, expected)

    def test_5_day_av(self):
        time_avgs = ['    5day 0hr 0min integral']
        result = _calc_integration_period(time_avgs)
        expected = [datetime.timedelta(0, (5*24*60*60))]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    tests.main()
