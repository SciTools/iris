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
Tests for `iris.fileformats.grib._load_convert.product_definition_template_0`.

"""
# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import datetime

import mock

from iris.fileformats.grib._load_convert import product_definition_template_0


MDI = -1
DATA_CUTOFF = 'data_cutoff'


class TestDataCutoff(tests.IrisTest):
    def _attributes(self, hours, minutes):
        section = {'hoursAfterDataCutoff': hours,
                   'minutesAfterDataCutoff': minutes,
                   'indicatorOfUnitOfTimeRange': 0,
                   'forecastTime': 0,
                   'NV': 0}
        metadata = {'attributes': {}, 'aux_coords_and_dims': []}
        frt_point = datetime.datetime(2014, 9, 23, 14, 45)
        product_definition_template_0(section, metadata, frt_point)
        return metadata['attributes']

    def test_none(self):
        attributes = self._attributes(MDI, MDI)
        self.assertNotIn(DATA_CUTOFF, attributes)

    def _check(self, hours, minutes, expected):
        attributes = self._attributes(hours, minutes)
        self.assertEqual(attributes, {DATA_CUTOFF: expected})

    def test_hours(self):
        self._check(3, MDI, datetime.datetime(2014, 9, 23, 17, 45))

    def test_minutes(self):
        self._check(MDI, 20, datetime.datetime(2014, 9, 23, 15, 5))

    def test_hours_and_minutes(self):
        self._check(30, 40, datetime.datetime(2014, 9, 24, 21, 25))


if __name__ == '__main__':
    tests.main()
