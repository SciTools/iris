# (C) British Crown Copyright 2014 - 2015, Met Office
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
Tests for `iris.fileformats.grib._load_convert.product_definition_template_31`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import range

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy
import warnings

import mock
import numpy as np

from iris.coords import AuxCoord
from iris.fileformats.grib._load_convert import product_definition_template_31


class Test(tests.IrisTest):
    def setUp(self):
        patch = mock.patch('warnings.warn')
        patch.start()
        self.addCleanup(patch.stop)
        self.metadata = {'factories': [], 'references': [],
                         'standard_name': None,
                         'long_name': None, 'units': None, 'attributes': None,
                         'cell_methods': [], 'dim_coords_and_dims': [],
                         'aux_coords_and_dims': []}

    def _check(self, request_warning=False, value=10, factor=1):
        # Prepare the arguments.
        series = mock.sentinel.satelliteSeries
        number = mock.sentinel.satelliteNumber
        instrument = mock.sentinel.instrumentType
        rt_coord = mock.sentinel.observation_time
        section = {'NB': 1,
                   'satelliteSeries': series,
                   'satelliteNumber': number,
                   'instrumentType': instrument,
                   'scaleFactorOfCentralWaveNumber': factor,
                   'scaledValueOfCentralWaveNumber': value}
        metadata = deepcopy(self.metadata)
        this = 'iris.fileformats.grib._load_convert.options'
        with mock.patch(this, warn_on_unsupported=request_warning):
            # The call being tested.
            product_definition_template_31(section, metadata, rt_coord)
        # Check the result.
        expected = deepcopy(self.metadata)
        coord = AuxCoord(series, long_name='satellite_series')
        expected['aux_coords_and_dims'].append((coord, None))
        coord = AuxCoord(number, long_name='satellite_number')
        expected['aux_coords_and_dims'].append((coord, None))
        coord = AuxCoord(instrument, long_name='instrument_type')
        expected['aux_coords_and_dims'].append((coord, None))
        unscale = lambda v, f: v / 10.0 ** f
        standard_name = 'sensor_band_central_radiation_wavenumber'
        coord = AuxCoord(unscale(value, factor),
                         standard_name=standard_name,
                         units='m-1')
        expected['aux_coords_and_dims'].append((coord, None))
        expected['aux_coords_and_dims'].append((rt_coord, None))
        self.assertEqual(metadata, expected)
        if request_warning:
            warn_msgs = [arg[1][0] for arg in warnings.warn.mock_calls]
            expected_msgs = ['type of generating process',
                             'observation generating process identifier']
            for emsg in expected_msgs:
                matches = [wmsg for wmsg in warn_msgs if emsg in wmsg]
                self.assertEqual(len(matches), 1)
                warn_msgs.remove(matches[0])
        else:
            self.assertEqual(len(warnings.warn.mock_calls), 0)

    def test_pdt_no_warn(self):
        self._check(request_warning=False)

    def test_pdt_warn(self):
        self._check(request_warning=True)

    def test_wavelength_array(self):
        value = np.array([1, 10, 100, 1000])
        for i in range(value.size):
            factor = np.ones(value.shape) * i
            self._check(value=value, factor=factor)


if __name__ == '__main__':
    tests.main()
