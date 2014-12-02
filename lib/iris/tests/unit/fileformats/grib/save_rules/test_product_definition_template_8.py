# (C) British Crown Copyright 2013 - 2014, Met Office
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
:func:`iris.fileformats.grib._save_rules.product_definition_template_8`

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock

from iris.coords import CellMethod, DimCoord
from iris.unit import Unit
import iris.tests.stock as stock
from iris.fileformats.grib._save_rules import product_definition_template_8


GRIB_API = 'iris.fileformats.grib._save_rules.gribapi'


class Test_typeOfStatisticalProcessing(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()
        # Rename cube to avoid warning about unknown discipline/parameter.
        self.cube.rename('air_temperature')
        coord = DimCoord(23, 'time', bounds=[0, 100],
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord)

    def test_sum(self):
        cube = self.cube
        cell_method = CellMethod(method='sum', coords=['time'])
        cube.add_cell_method(cell_method)

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            product_definition_template_8(cube, grib)
        mock_gribapi.assert_has_calls(mock.call.grib_set_long(
            grib, "typeOfStatisticalProcessing", 1))

    def test_unrecognised(self):
        cube = self.cube
        cell_method = CellMethod(method='95th percentile', coords=['time'])
        cube.add_cell_method(cell_method)

        grib = mock.sentinel.grib_msg_id
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            product_definition_template_8(cube, grib)
        mock_gribapi.assert_has_calls(mock.call.grib_set_long(
            grib, "typeOfStatisticalProcessing", 255))

    def test_multiple_cell_method_coords(self):
        cube = self.cube
        cell_method = CellMethod(method='sum',
                                 coords=['time', 'forecast_period'])
        cube.add_cell_method(cell_method)

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            with self.assertRaisesRegexp(
                    ValueError, 'Cannot handle multiple coordinate name'):
                product_definition_template_8(cube, grib)

    def test_cell_method_coord_name_fail(self):
        cube = self.cube
        cell_method = CellMethod(method='mean', coords=['season'])
        cube.add_cell_method(cell_method)

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            with self.assertRaisesRegexp(
                    ValueError, "Expected a cell method with a coordinate "
                    "name of 'time'"):
                product_definition_template_8(cube, grib)


class Test_TimeCoordPrerequisites(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()
        # Rename cube to avoid warning about unknown discipline/parameter.
        self.cube.rename('air_temperature')

    def test_multiple_points(self):
        # Add time coord with multiple points.
        coord = DimCoord([23, 24, 25], 'time',
                         bounds=[[22, 23], [23, 24], [24, 25]],
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord, 0)
        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            with self.assertRaisesRegexp(
                    ValueError, 'Expected length one time coordinate'):
                product_definition_template_8(self.cube, grib)

    def test_no_bounds(self):
        # Add time coord with no bounds.
        coord = DimCoord(23, 'time',
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord)
        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            with self.assertRaisesRegexp(
                    ValueError, 'Expected time coordinate with two bounds, '
                    'got 0 bounds'):
                product_definition_template_8(self.cube, grib)

    def test_more_than_two_bounds(self):
        # Add time coord with more than two bounds.
        coord = DimCoord(23, 'time', bounds=[21, 22, 23],
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord)
        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            with self.assertRaisesRegexp(
                    ValueError, 'Expected time coordinate with two bounds, '
                    'got 3 bounds'):
                product_definition_template_8(self.cube, grib)


if __name__ == "__main__":
    tests.main()
