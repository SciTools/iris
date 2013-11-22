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
import numpy as np

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


if __name__ == "__main__":
    tests.main()
