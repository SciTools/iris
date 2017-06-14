# (C) British Crown Copyright 2014 - 2017, Met Office
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
Tests for
:func:`iris.fileformats.grib._load_convert.product_definition_template_31`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris.tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.fileformats.grib._load_convert import product_definition_template_31
from iris.tests import mock
from iris.tests.unit.fileformats.grib.load_convert import empty_metadata


class Test(tests.IrisTest):
    def setUp(self):
        self.patch('warnings.warn')
        self.satellite_common_patch = self.patch(
            'iris.fileformats.grib._load_convert.satellite_common')
        self.generating_process_patch = self.patch(
            'iris.fileformats.grib._load_convert.generating_process')

    def test(self):
        # Prepare the arguments.
        series = mock.sentinel.satelliteSeries
        number = mock.sentinel.satelliteNumber
        instrument = mock.sentinel.instrumentType
        rt_coord = mock.sentinel.observation_time
        section = {'NB': 1,
                   'satelliteSeries': series,
                   'satelliteNumber': number,
                   'instrumentType': instrument,
                   'scaleFactorOfCentralWaveNumber': 1,
                   'scaledValueOfCentralWaveNumber': 12}

        # Call the function.
        metadata = empty_metadata()
        product_definition_template_31(section, metadata, rt_coord)

        # Check that 'satellite_common' was called.
        self.assertEqual(self.satellite_common_patch.call_count, 1)
        # Check that 'generating_process' was called.
        self.assertEqual(self.generating_process_patch.call_count, 1)
        # Check that the scalar time coord was added in.
        self.assertIn((rt_coord, None), metadata['aux_coords_and_dims'])


if __name__ == '__main__':
    tests.main()
