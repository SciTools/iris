# (C) British Crown Copyright 2013, Met Office
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

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock

import iris.fileformats.grib
from iris.fileformats.grib.grib_save_rules import reference_time
import iris.tests.stock as stock
from iris.tests.test_grib_load import TestGribSimple


class Test(TestGribSimple):
    def test_forecast_period(self):
        # The stock cube has a non-compliant forecast_period.
        iris.fileformats.grib.hindcast_workaround = True
        cube = stock.global_grib2()
        cube.coord("forecast_period").points = [36]

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch('iris.fileformats.grib.grib_save_rules.gribapi',
                        mock_gribapi):
            reference_time(cube, grib)

        mock_gribapi.assert_has_calls(
            [mock.call.grib_set_long(grib, "significanceOfReferenceTime", 1),
             mock.call.grib_set_long(grib, "dataDate", '19941129'),
             mock.call.grib_set_long(grib, "dataTime", '1200')])

    def test_no_forecast_period(self):
        # The stock cube has a non-compliant forecast_period.
        iris.fileformats.grib.hindcast_workaround = True
        cube = stock.global_grib2()
        cube.remove_coord("forecast_period")

        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch('iris.fileformats.grib.grib_save_rules.gribapi',
                        mock_gribapi):
            reference_time(cube, grib)

        mock_gribapi.assert_has_calls(
            [mock.call.grib_set_long(grib, "significanceOfReferenceTime", 2),
             mock.call.grib_set_long(grib, "dataDate", '19941201'),
             mock.call.grib_set_long(grib, "dataTime", '0000')])

if __name__ == "__main__":
    tests.main()
