# (C) British Crown Copyright 2017, Met Office
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
:func:`iris.fileformats.grib._load_convert.product_definition_template_15`.

This basically copies code from 'test_product_definition_template_0', and adds
testing for the statistical method and spatial-processing type.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris.tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.coords import CellMethod, DimCoord
from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import product_definition_template_15
from iris.fileformats.grib._load_convert import _MDI as MDI
from iris.tests.unit.fileformats.grib.load_convert import (LoadConvertTest,
                                                           empty_metadata)


def section_4():
    return {'productDefinitionTemplateNumber': 15,
            'hoursAfterDataCutoff': MDI,
            'minutesAfterDataCutoff': MDI,
            'indicatorOfUnitOfTimeRange': 0,  # minutes
            'forecastTime': 360,
            'NV': 0,
            'typeOfFirstFixedSurface': 103,
            'scaleFactorOfFirstFixedSurface': 0,
            'scaledValueOfFirstFixedSurface': 9999,
            'typeOfSecondFixedSurface': 255,
            'statisticalProcess': 2,  # method = maximum
            'spatialProcessing': 0,  # from source grid, no interpolation
            'numberOfPointsUsed': 0  # unused?
            }


class Test(LoadConvertTest):
    def setUp(self):
        self.ref_time_coord = DimCoord(24, 'time', units='hours since epoch')

    def _check_translate(self, section):
        metadata = empty_metadata()
        product_definition_template_15(section, metadata,
                                       self.ref_time_coord)
        return metadata

    def test_t(self):
        metadata = self._check_translate(section_4())

        expected = empty_metadata()
        aux = expected['aux_coords_and_dims']
        aux.append((DimCoord(6, 'forecast_period', units='hours'), None))
        aux.append((DimCoord(18, 'forecast_reference_time',
                             units='hours since epoch'), None))
        aux.append((self.ref_time_coord, None))
        aux.append((DimCoord(9999, long_name='height', units='m'),
                    None))
        expected['cell_methods'] = [CellMethod(coords=('area',),
                                               method='maximum')]

        self.assertMetadataEqual(metadata, expected)

    def test_bad_statistic_method(self):
        section = section_4()
        section['statisticalProcess'] = 999
        msg = ('unsupported statistical process type \[999\]')
        with self.assertRaisesRegexp(TranslationError, msg):
            self._check_translate(section)

    def test_bad_spatial_processing_code(self):
        section = section_4()
        section['spatialProcessing'] = 999
        msg = ('unsupported spatial processing type \[999\]')
        with self.assertRaisesRegexp(TranslationError, msg):
            self._check_translate(section)


if __name__ == '__main__':
    tests.main()
