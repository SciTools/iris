# (C) British Crown Copyright 2013 - 2017, Met Office
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
"""Unit tests for `iris.fileformats.grib.grib_save_rules.reference_time`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock

from iris.fileformats.grib import load_cubes
from iris.fileformats.grib._save_rules import reference_time


class Test(tests.IrisTest):
    def _test(self, cube):
        grib = mock.Mock()
        mock_gribapi = mock.Mock(spec=gribapi)
        with mock.patch('iris.fileformats.grib._save_rules.gribapi',
                        mock_gribapi):
            reference_time(cube, grib)

        mock_gribapi.assert_has_calls(
            [mock.call.grib_set_long(grib, "significanceOfReferenceTime", 1),
             mock.call.grib_set_long(grib, "dataDate", '19980306'),
             mock.call.grib_set_long(grib, "dataTime", '0300')])

    @tests.skip_data
    def test_forecast_period(self):
        # The stock cube has a non-compliant forecast_period.
        fname = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
        [cube] = load_cubes(fname)
        self._test(cube)

    @tests.skip_data
    def test_no_forecast_period(self):
        # The stock cube has a non-compliant forecast_period.
        fname = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
        [cube] = load_cubes(fname)
        cube.remove_coord("forecast_period")
        self._test(cube)


if __name__ == "__main__":
    tests.main()
