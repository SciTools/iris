# (C) British Crown Copyright 2014 - 2017, Met Office
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
Tests for `iris_grib._load_convert.satellite_common`.

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

from iris.coords import AuxCoord
from iris_grib.tests.unit.load_convert import empty_metadata

from iris_grib._load_convert import satellite_common


class Test(tests.IrisGribTest):
    def _check(self, factors=1, values=111):
        # Prepare the arguments.
        series = mock.sentinel.satelliteSeries
        number = mock.sentinel.satelliteNumber
        instrument = mock.sentinel.instrumentType
        section = {'NB': 1,
                   'satelliteSeries': series,
                   'satelliteNumber': number,
                   'instrumentType': instrument,
                   'scaleFactorOfCentralWaveNumber': factors,
                   'scaledValueOfCentralWaveNumber': values}

        # Call the function.
        metadata = empty_metadata()
        satellite_common(section, metadata)

        # Check the result.
        expected = empty_metadata()
        coord = AuxCoord(series, long_name='satellite_series')
        expected['aux_coords_and_dims'].append((coord, None))
        coord = AuxCoord(number, long_name='satellite_number')
        expected['aux_coords_and_dims'].append((coord, None))
        coord = AuxCoord(instrument, long_name='instrument_type')
        expected['aux_coords_and_dims'].append((coord, None))
        standard_name = 'sensor_band_central_radiation_wavenumber'
        coord = AuxCoord(values / (10.0 ** factors),
                         standard_name=standard_name,
                         units='m-1')
        expected['aux_coords_and_dims'].append((coord, None))
        self.assertEqual(metadata, expected)

    def test_basic(self):
        self._check()

    def test_multiple_wavelengths(self):
        # Check with multiple values, and several different scaling factors.
        values = np.array([1, 11, 123, 1975])
        for i_factor in (-3, -1, 0, 1, 3):
            factors = np.ones(values.shape) * i_factor
            self._check(values=values, factors=factors)


if __name__ == '__main__':
    tests.main()
