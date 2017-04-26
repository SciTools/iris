# (C) British Crown Copyright 2017, Met Office
#
# This file is part of iris-grib.
#
# iris-grib is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# iris-grib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with iris-grib.  If not, see <http://www.gnu.org/licenses/>.
"""
Tests for `iris_grib._load_convert.product_definition_template_32`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris_grib.tests first so that some things can be initialised
# before importing anything else.
import iris_grib.tests as tests

from copy import deepcopy
import mock
import numpy as np
import warnings

from iris.coords import AuxCoord, DimCoord
from iris_grib.tests.unit.load_convert import empty_metadata

from iris_grib._load_convert import product_definition_template_32

MDI = 0xffffffff


class Test(tests.IrisGribTest):
    def setUp(self):
        self.patch('warnings.warn')
        self.generating_process_patch = self.patch(
            'iris_grib._load_convert.generating_process')
        self.satellite_common_patch = self.patch(
            'iris_grib._load_convert.satellite_common')
        self.time_coords_patch = self.patch(
            'iris_grib._load_convert.time_coords')
        self.data_cutoff_patch = self.patch(
            'iris_grib._load_convert.data_cutoff')

    def test(self, value=10, factor=1):
        # Prepare the arguments.
        series = mock.sentinel.satelliteSeries
        number = mock.sentinel.satelliteNumber
        instrument = mock.sentinel.instrumentType
        rt_coord = mock.sentinel.observation_time
        section = {'NB': 1,
                   'hoursAfterDataCutoff': None,
                   'minutesAfterDataCutoff': None,
                   'satelliteSeries': series,
                   'satelliteNumber': number,
                   'instrumentType': instrument,
                   'scaleFactorOfCentralWaveNumber': 1,
                   'scaledValueOfCentralWaveNumber': 12,
                   }

        # Call the function.
        metadata = empty_metadata()
        product_definition_template_32(section, metadata, rt_coord)

        # Check that 'satellite_common' was called.
        self.assertEqual(self.satellite_common_patch.call_count, 1)
        # Check that 'generating_process' was called.
        self.assertEqual(self.generating_process_patch.call_count, 1)
        # Check that 'data_cutoff' was called.
        self.assertEqual(self.data_cutoff_patch.call_count, 1)
        # Check that 'time_coords' was called.
        self.assertEqual(self.time_coords_patch.call_count, 1)


if __name__ == '__main__':
    tests.main()
