# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Integration tests for loading and saving GRIB2 files."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy.ma as ma

from iris import FUTURE, load_cube

from subprocess import check_output

import iris
from iris import FUTURE, load_cube, save
from iris.coords import CellMethod
from iris.coord_systems import RotatedGeogCS
from iris.fileformats.pp import EARTH_RADIUS as UM_DEFAULT_EARTH_RADIUS
from iris.util import is_regular


@tests.skip_data
class TestLoad(tests.IrisTest):
    def test_bulletin(self):
        with FUTURE.context(strict_grib_load=False):
            path = tests.get_data_path(('GRIB', 'time_processed',
                                        'time_bound.grib1'))
            cube = load_cube(path)
            self.assertEqual(cube.long_name, 'UNKNOWN LOCAL PARAM 106.137')
            self.assertEqual(cube.shape, (321, 481))
            self.assertLess(cube.data.min(), 5.1e-6)
            self.assertGreater(cube.data.min(), 5.0e-6)

    def test_bulletin_strict(self):
        with FUTURE.context(strict_grib_load=True):
            path = tests.get_data_path(('GRIB', 'time_processed',
                                        'time_bound.grib1'))
            cube = load_cube(path)
            self.assertEqual(cube.long_name, 'UNKNOWN LOCAL PARAM 106.137')
            self.assertEqual(cube.shape, (321, 481))
            self.assertLess(cube.data.min(), 5.1e-6)
            self.assertGreater(cube.data.min(), 5.0e-6)


if __name__ == '__main__':
    tests.main()
