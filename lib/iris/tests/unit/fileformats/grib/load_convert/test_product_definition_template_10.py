# (C) British Crown Copyright 2016 - 2017, Met Office
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
Test function
:func:`iris.fileformats.grib._load_convert.product_definition_template_10`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris.tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
import mock

from iris.coords import DimCoord
from iris.fileformats.grib._load_convert import product_definition_template_10
from iris.tests.unit.fileformats.grib.load_convert import empty_metadata


class Test(tests.IrisGribTest):
    def setUp(self):
        module = 'iris.fileformats.grib._load_convert'
        this_module = '{}.product_definition_template_10'.format(module)
        self.patch_statistical_fp_coord = self.patch(
            module + '.statistical_forecast_period_coord',
            return_value=mock.sentinel.dummy_fp_coord)
        self.patch_time_coord = self.patch(
            module + '.validity_time_coord',
            return_value=mock.sentinel.dummy_time_coord)
        self.patch_vertical_coords = self.patch(module + '.vertical_coords')

    def test_percentile_coord(self):
        metadata = empty_metadata()
        percentileValue = 75
        section = {'percentileValue': percentileValue,
                   'hoursAfterDataCutoff': 1,
                   'minutesAfterDataCutoff': 1,
                   'numberOfTimeRange': 1,
                   'typeOfStatisticalProcessing': 1,
                   'typeOfTimeIncrement': 2,
                   'timeIncrement': 0,
                   'yearOfEndOfOverallTimeInterval': 2000,
                   'monthOfEndOfOverallTimeInterval': 1,
                   'dayOfEndOfOverallTimeInterval': 1,
                   'hourOfEndOfOverallTimeInterval': 1,
                   'minuteOfEndOfOverallTimeInterval': 0,
                   'secondOfEndOfOverallTimeInterval': 1}
        forecast_reference_time = mock.Mock()
        # The called being tested.
        product_definition_template_10(section, metadata,
                                       forecast_reference_time)

        expected = {'aux_coords_and_dims': []}
        percentile = DimCoord(percentileValue,
                              long_name='percentile_over_time',
                              units='no_unit')
        expected['aux_coords_and_dims'].append((percentile, None))

        self.assertEqual(metadata['aux_coords_and_dims'][-1],
                         expected['aux_coords_and_dims'][0])


if __name__ == '__main__':
    tests.main()
