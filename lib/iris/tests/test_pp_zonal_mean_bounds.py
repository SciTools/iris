# (C) British Crown Copyright 2010 - 2017, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import os

import iris
import iris.util


@tests.skip_data
class TestZonalMeanBounds(tests.IrisTest):
    def test_mulitple_longitude(self):
        # test that bounds are set for a zonal mean file with many longitude
        # values
        orig_file = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))

        f = next(iris.fileformats.pp.load(orig_file))
        f.lbproc = 192  # time and zonal mean

        # Write out pp file
        temp_filename = iris.util.create_temp_filename(".pp")
        with open(temp_filename, 'wb') as temp_fh:
            f.save(temp_fh)

        # Load pp file
        cube = iris.load_cube(temp_filename)

        self.assertTrue(cube.coord('longitude').has_bounds())

        os.remove(temp_filename)

    def test_singular_longitude(self):
        # test that bounds are set for a zonal mean file with a single longitude
        # value

        pp_file = tests.get_data_path(('PP', 'zonal_mean', 'zonal_mean.pp'))

        # Load pp file
        cube = iris.load_cube(pp_file)

        self.assertTrue(cube.coord('longitude').has_bounds())


if __name__ == "__main__":
    tests.main()
