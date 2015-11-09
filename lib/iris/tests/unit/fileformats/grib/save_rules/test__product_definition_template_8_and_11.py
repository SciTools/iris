# (C) British Crown Copyright 2013 - 2015, Met Office
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
:func:`iris.fileformats.grib._save_rules._product_definition_template_8_and_11`

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi

from iris.coords import CellMethod, DimCoord
from cf_units import Unit
from iris.tests import mock
import iris.tests.stock as stock
from iris.fileformats.grib._save_rules import \
    _product_definition_template_8_and_11


class TestTypeOfStatisticalProcessing(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()
        # Rename cube to avoid warning about unknown discipline/parameter.
        self.cube.rename('air_temperature')
        coord = DimCoord(23, 'time', bounds=[0, 100],
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord)

    @mock.patch.object(gribapi, 'grib_set')
    def test_sum(self, mock_set):
        cube = self.cube
        cell_method = CellMethod(method='sum', coords=['time'])
        cube.add_cell_method(cell_method)

        _product_definition_template_8_and_11(cube, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 "typeOfStatisticalProcessing", 1)

    @mock.patch.object(gribapi, 'grib_set')
    def test_unrecognised(self, mock_set):
        cube = self.cube
        cell_method = CellMethod(method='95th percentile', coords=['time'])
        cube.add_cell_method(cell_method)

        _product_definition_template_8_and_11(cube, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib,
                                 "typeOfStatisticalProcessing", 255)

    @mock.patch.object(gribapi, 'grib_set')
    def test_multiple_cell_method_coords(self, mock_set):
        cube = self.cube
        cell_method = CellMethod(method='sum',
                                 coords=['time', 'forecast_period'])
        cube.add_cell_method(cell_method)
        with self.assertRaisesRegexp(ValueError,
                                     'Cannot handle multiple coordinate name'):
            _product_definition_template_8_and_11(cube, mock.sentinel.grib)

    @mock.patch.object(gribapi, 'grib_set')
    def test_cell_method_coord_name_fail(self, mock_set):
        cube = self.cube
        cell_method = CellMethod(method='mean', coords=['season'])
        cube.add_cell_method(cell_method)
        with self.assertRaisesRegexp(
                ValueError, "Expected a cell method with a coordinate "
                "name of 'time'"):
            _product_definition_template_8_and_11(cube, mock.sentinel.grib)


class TestTimeCoordPrerequisites(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()
        # Rename cube to avoid warning about unknown discipline/parameter.
        self.cube.rename('air_temperature')

    @mock.patch.object(gribapi, 'grib_set')
    def test_multiple_points(self, mock_set):
        # Add time coord with multiple points.
        coord = DimCoord([23, 24, 25], 'time',
                         bounds=[[22, 23], [23, 24], [24, 25]],
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord, 0)
        with self.assertRaisesRegexp(
                ValueError, 'Expected length one time coordinate'):
            _product_definition_template_8_and_11(self.cube,
                                                  mock.sentinel.grib)

    @mock.patch.object(gribapi, 'grib_set')
    def test_no_bounds(self, mock_set):
        # Add time coord with no bounds.
        coord = DimCoord(23, 'time',
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord)
        with self.assertRaisesRegexp(
                ValueError, 'Expected time coordinate with two bounds, '
                'got 0 bounds'):
            _product_definition_template_8_and_11(self.cube,
                                                  mock.sentinel.grib)

    @mock.patch.object(gribapi, 'grib_set')
    def test_more_than_two_bounds(self, mock_set):
        # Add time coord with more than two bounds.
        coord = DimCoord(23, 'time', bounds=[21, 22, 23],
                         units=Unit('days since epoch', calendar='standard'))
        self.cube.add_aux_coord(coord)
        with self.assertRaisesRegexp(
                ValueError, 'Expected time coordinate with two bounds, '
                'got 3 bounds'):
            _product_definition_template_8_and_11(self.cube,
                                                  mock.sentinel.grib)


class TestEndOfOverallTimeInterval(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()
        # Rename cube to avoid warning about unknown discipline/parameter.
        self.cube.rename('air_temperature')
        cell_method = CellMethod(method='sum', coords=['time'])
        self.cube.add_cell_method(cell_method)

    @mock.patch.object(gribapi, 'grib_set')
    def test_default_calendar(self, mock_set):
        cube = self.cube
        # End bound is 1972-04-26 10:27:07.
        coord = DimCoord(23.0, 'time', bounds=[0.452, 20314.452],
                         units=Unit('hours since epoch'))
        cube.add_aux_coord(coord)

        grib = mock.sentinel.grib
        _product_definition_template_8_and_11(cube, grib)

        mock_set.assert_any_call(
            grib, "yearOfEndOfOverallTimeInterval", 1972)
        mock_set.assert_any_call(
            grib, "monthOfEndOfOverallTimeInterval", 4)
        mock_set.assert_any_call(
            grib, "dayOfEndOfOverallTimeInterval", 26)
        mock_set.assert_any_call(
            grib, "hourOfEndOfOverallTimeInterval", 10)
        mock_set.assert_any_call(
            grib, "minuteOfEndOfOverallTimeInterval", 27)
        mock_set.assert_any_call(
            grib, "secondOfEndOfOverallTimeInterval", 7)

    @mock.patch.object(gribapi, 'grib_set')
    def test_360_day_calendar(self, mock_set):
        cube = self.cube
        # End bound is 1972-05-07 10:27:07
        coord = DimCoord(23.0, 'time', bounds=[0.452, 20314.452],
                         units=Unit('hours since epoch', calendar='360_day'))
        cube.add_aux_coord(coord)

        grib = mock.sentinel.grib
        _product_definition_template_8_and_11(cube, grib)

        mock_set.assert_any_call(
            grib, "yearOfEndOfOverallTimeInterval", 1972)
        mock_set.assert_any_call(
            grib, "monthOfEndOfOverallTimeInterval", 5)
        mock_set.assert_any_call(
            grib, "dayOfEndOfOverallTimeInterval", 7)
        mock_set.assert_any_call(
            grib, "hourOfEndOfOverallTimeInterval", 10)
        mock_set.assert_any_call(
            grib, "minuteOfEndOfOverallTimeInterval", 27)
        mock_set.assert_any_call(
            grib, "secondOfEndOfOverallTimeInterval", 7)


class TestNumberOfTimeRange(tests.IrisTest):
    @mock.patch.object(gribapi, 'grib_set')
    def test_other_cell_methods(self, mock_set):
        cube = stock.lat_lon_cube()
        # Rename cube to avoid warning about unknown discipline/parameter.
        cube.rename('air_temperature')
        coord = DimCoord(23, 'time', bounds=[0, 24],
                         units=Unit('hours since epoch'))
        cube.add_aux_coord(coord)
        # Add one time cell method and another unrelated one.
        cell_method = CellMethod(method='mean', coords=['elephants'])
        cube.add_cell_method(cell_method)
        cell_method = CellMethod(method='sum', coords=['time'])
        cube.add_cell_method(cell_method)

        _product_definition_template_8_and_11(cube, mock.sentinel.grib)
        mock_set.assert_any_call(mock.sentinel.grib, 'numberOfTimeRange', 1)


if __name__ == "__main__":
    tests.main()
