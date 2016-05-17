# (C) British Crown Copyright 2016, Met Office
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
:func:`iris.fileformats.grib._load_convert.product_definition_template_40`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import iris.coords
from iris.fileformats.grib._load_convert import product_definition_template_40
from iris.tests.unit.fileformats.grib.load_convert import empty_metadata


MDI = 0xffffffff


def section_4():
    return {'hoursAfterDataCutoff': MDI,
            'minutesAfterDataCutoff': MDI,
            'constituentType': 1,
            'indicatorOfUnitOfTimeRange': 0,  # minutes
            'startStep': 360,
            'NV': 0,
            'typeOfFirstFixedSurface': 103,
            'scaleFactorOfFirstFixedSurface': 0,
            'scaledValueOfFirstFixedSurface': 9999,
            'typeOfSecondFixedSurface': 255}


class Test(tests.IrisTest):
    def test_constituent_type(self):
        metadata = empty_metadata()
        rt_coord = iris.coords.DimCoord(24, 'forecast_reference_time',
                                        units='hours since epoch')
        product_definition_template_40(section_4(), metadata, rt_coord)
        expected = empty_metadata()
        expected['attributes']['WMO_constituent_type'] = 1
        self.assertEqual(metadata['attributes'], expected['attributes'])


if __name__ == '__main__':
    tests.main()
