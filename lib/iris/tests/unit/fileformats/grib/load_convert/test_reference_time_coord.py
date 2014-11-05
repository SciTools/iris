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
Test function :func:`iris.fileformats.grib._load_convert.reference_time_coord.

Reference Code Table 1.2.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
from datetime import datetime

from iris.coords import DimCoord
from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import reference_time_coord
from iris.unit import CALENDAR_GREGORIAN, Unit


class Test(tests.IrisTest):
    def setUp(self):
        self.section = {'year': 2007,
                        'month': 1,
                        'day': 15,
                        'hour': 0,
                        'minute': 3,
                        'second': 0}
        self.unit = Unit('hours since epoch', calendar=CALENDAR_GREGORIAN)
        dt = datetime(self.section['year'], self.section['month'],
                      self.section['day'], self.section['hour'],
                      self.section['minute'], self.section['second'])
        self.point = self.unit.date2num(dt)

    def _check(self, section, standard_name=None):
        expected = DimCoord(self.point, standard_name=standard_name,
                            units=self.unit)
        # The call being tested.
        coord = reference_time_coord(section)
        self.assertEqual(coord, expected)

    def test_start_of_forecast(self):
        section = deepcopy(self.section)
        section['significanceOfReferenceTime'] = 1
        self._check(section, 'forecast_reference_time')

    def test_observation_time(self):
        section = deepcopy(self.section)
        section['significanceOfReferenceTime'] = 3
        self._check(section, 'time')

    def test_unknown_significance(self):
        section = deepcopy(self.section)
        section['significanceOfReferenceTime'] = 0
        emsg = 'unsupported significance'
        with self.assertRaisesRegexp(TranslationError, emsg):
            self._check(section)


if __name__ == '__main__':
    tests.main()
