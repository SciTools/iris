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
Unit tests for
:func:`iris.fileformats.grib._save_rules.product_definition_template_1`

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from cf_units import Unit
import gribapi
import mock

from iris.coords import CellMethod, DimCoord
import iris.tests.stock as stock

from iris.fileformats.grib._save_rules import product_definition_template_1


class TestRealizationIdentifier(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()
        # Rename cube to avoid warning about unknown discipline/parameter.
        self.cube.rename('air_temperature')
        coord = DimCoord([45], 'time',
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord)

    @mock.patch.object(gribapi, 'grib_set')
    def test_realization(self, mock_set):
        cube = self.cube
        coord = DimCoord(10, 'realization', units='1')
        cube.add_aux_coord(coord)

        product_definition_template_1(cube, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 "productDefinitionTemplateNumber", 1)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 "perturbationNumber", 10)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 "numberOfForecastsInEnsemble", 255)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 "typeOfEnsembleForecast", 255)

    @mock.patch.object(gribapi, 'grib_set')
    def test_multiple_realization_values(self, mock_set):
        cube = self.cube
        coord = DimCoord([8, 9, 10], 'realization', units='1')
        cube.add_aux_coord(coord, 0)

        msg = "'realization' coordinate with one point is required"
        with self.assertRaisesRegexp(ValueError, msg):
            product_definition_template_1(cube, mock.sentinel.grib)


if __name__ == "__main__":
    tests.main()
