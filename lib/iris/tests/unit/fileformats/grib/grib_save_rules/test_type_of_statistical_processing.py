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
"""Unit tests for module-level functions."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock

from iris.fileformats.grib.grib_save_rules \
    import type_of_statistical_processing

from iris.tests.test_grib_load import TestGribSimple


class Test(TestGribSimple):
    def test_sum(self):
        cube = mock.Mock()
        cube.cell_methods = [mock.Mock(method='sum', coord_names=['ni'])]

        coord = mock.Mock()
        coord.name = mock.Mock(return_value='ni')

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch('iris.fileformats.grib.grib_save_rules.gribapi',
                        mock_gribapi):
            type_of_statistical_processing(cube, grib, coord)

        mock_gribapi.assert_has_calls(mock.call.grib_set_long(
            grib, "typeOfStatisticalProcessing", 1))

    def test_missing(self):
        cube = mock.Mock()
        cube.cell_methods = [mock.Mock(method='95th percentile',
                                       coord_names=['time'])]

        coord = mock.Mock()
        coord.name = mock.Mock(return_value='time')

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch('iris.fileformats.grib.grib_save_rules.gribapi',
                        mock_gribapi):
            type_of_statistical_processing(cube, grib, coord)

        mock_gribapi.assert_has_calls(mock.call.grib_set_long(
            grib, "typeOfStatisticalProcessing", 255))

    def test_cell_method_coords_len_fail(self):
        cube = mock.Mock()
        cube.cell_methods = [mock.Mock(method='sum', coord_names=['time',
                                                                  'fp'])]

        coord = mock.Mock()
        coord.name = mock.Mock(return_value='time')

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch('iris.fileformats.grib.grib_save_rules.gribapi',
                        mock_gribapi):
            with self.assertRaisesRegexp(ValueError,
                                         'There are multiple coord names '
                                         'referenced by the primary cell '
                                         'method:'):
                type_of_statistical_processing(cube, grib, coord)

    def test_cell_method_coord_name_fail(self):
        cube = mock.Mock()
        cube.cell_methods = [mock.Mock(method='sum', coord_names=['time'])]

        coord = mock.Mock()
        coord.name = mock.Mock(return_value='forecast_period')

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch('iris.fileformats.grib.grib_save_rules.gribapi',
                        mock_gribapi):
            with self.assertRaisesRegexp(ValueError,
                                         'The coord name referenced by the '
                                         'primary cell method'):
                type_of_statistical_processing(cube, grib, coord)


if __name__ == "__main__":
    tests.main()
