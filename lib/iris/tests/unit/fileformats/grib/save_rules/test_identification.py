# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for `iris.fileformats.grib.grib_save_rules.identification`."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock

import iris.fileformats.grib
from iris.fileformats.grib._save_rules import identification
import iris.tests.stock as stock
from iris.tests.test_grib_load import TestGribSimple


GRIB_API = 'iris.fileformats.grib._save_rules.gribapi'


class Test(TestGribSimple):
    @tests.skip_data
    def test_no_realization(self):
        cube = stock.simple_pp()
        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            identification(cube, grib)

        mock_gribapi.assert_has_calls(
            [mock.call.grib_set_long(grib, "typeOfProcessedData", 2)])

    @tests.skip_data
    def test_realization_0(self):
        cube = stock.simple_pp()
        realisation = iris.coords.AuxCoord((0,), standard_name='realization',
                                           units='1')
        cube.add_aux_coord(realisation)

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            identification(cube, grib)

        mock_gribapi.assert_has_calls(
            [mock.call.grib_set_long(grib, "typeOfProcessedData", 3)])

    @tests.skip_data
    def test_realization_n(self):
        cube = stock.simple_pp()
        realisation = iris.coords.AuxCoord((2,), standard_name='realization',
                                           units='1')
        cube.add_aux_coord(realisation)

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch(GRIB_API, mock_gribapi):
            identification(cube, grib)

        mock_gribapi.assert_has_calls(
            [mock.call.grib_set_long(grib, "typeOfProcessedData", 4)])


if __name__ == "__main__":
    tests.main()
