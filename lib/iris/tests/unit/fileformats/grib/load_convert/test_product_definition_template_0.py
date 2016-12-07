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
Test function
:func:`iris.fileformats.grib._load_convert.product_definition_template_0`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import iris.coords
from iris.tests.unit.fileformats.grib.load_convert import (LoadConvertTest,
                                                           empty_metadata)
from iris.fileformats.grib._load_convert import product_definition_template_0
from iris.tests import mock


MDI = 0xffffffff


def section_4():
    return {'hoursAfterDataCutoff': MDI,
            'minutesAfterDataCutoff': MDI,
            'indicatorOfUnitOfTimeRange': 0,  # minutes
            'forecastTime': 360,
            'NV': 0,
            'typeOfFirstFixedSurface': 103,
            'scaleFactorOfFirstFixedSurface': 0,
            'scaledValueOfFirstFixedSurface': 9999,
            'typeOfSecondFixedSurface': 255}


class Test(LoadConvertTest):
    def test_given_frt(self):
        metadata = empty_metadata()
        rt_coord = iris.coords.DimCoord(24, 'forecast_reference_time',
                                        units='hours since epoch')
        product_definition_template_0(section_4(), metadata, rt_coord)
        expected = empty_metadata()
        aux = expected['aux_coords_and_dims']
        aux.append((iris.coords.DimCoord(6, 'forecast_period', units='hours'),
                    None))
        aux.append((
            iris.coords.DimCoord(30, 'time', units='hours since epoch'), None))
        aux.append((rt_coord, None))
        aux.append((iris.coords.DimCoord(9999, long_name='height', units='m'),
                    None))
        self.assertMetadataEqual(metadata, expected)

    def test_given_t(self):
        metadata = empty_metadata()
        rt_coord = iris.coords.DimCoord(24, 'time',
                                        units='hours since epoch')
        product_definition_template_0(section_4(), metadata, rt_coord)
        expected = empty_metadata()
        aux = expected['aux_coords_and_dims']
        aux.append((iris.coords.DimCoord(6, 'forecast_period', units='hours'),
                    None))
        aux.append((
            iris.coords.DimCoord(18, 'forecast_reference_time',
                                 units='hours since epoch'), None))
        aux.append((rt_coord, None))
        aux.append((iris.coords.DimCoord(9999, long_name='height', units='m'),
                    None))
        self.assertMetadataEqual(metadata, expected)

    def test_generating_process_warnings(self):
        metadata = empty_metadata()
        rt_coord = iris.coords.DimCoord(24, 'forecast_reference_time',
                                        units='hours since epoch')
        convert_options = iris.fileformats.grib._load_convert.options
        emit_warnings = convert_options.warn_on_unsupported
        try:
            convert_options.warn_on_unsupported = True
            with mock.patch('warnings.warn') as warn:
                product_definition_template_0(section_4(), metadata, rt_coord)
            warn_msgs = [call[1][0] for call in warn.mock_calls]
            expected = ['Unable to translate type of generating process.',
                        'Unable to translate background generating process '
                        'identifier.',
                        'Unable to translate forecast generating process '
                        'identifier.']
            self.assertEqual(warn_msgs, expected)
        finally:
            convert_options.warn_on_unsupported = emit_warnings


if __name__ == '__main__':
    tests.main()
