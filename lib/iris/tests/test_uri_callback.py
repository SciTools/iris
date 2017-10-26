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

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os

import iris.coords


@tests.skip_data
class TestCallbacks(tests.IrisTest):
    @tests.skip_grib
    def test_grib_callback(self):
        def grib_thing_getter(cube, field, filename):
            if hasattr(field, 'sections'):
                # New-style loader callback : 'field' is a GribMessage, which has 'sections'.
                cube.add_aux_coord(iris.coords.AuxCoord(field.sections[1]['year'], long_name='extra_year_number_coord', units='no_unit'))
            else:
                # Old-style loader provides 'GribWrapper' type field.
                cube.add_aux_coord(iris.coords.AuxCoord(field.extra_keys['_periodStartDateTime'], long_name='random element', units='no_unit'))

        fname = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
        cube = iris.load_cube(fname, callback=grib_thing_getter)
        self.assertCML(cube, ['uri_callback', 'grib_global.cml'])

    def test_pp_callback(self):
        def pp_callback(cube, field, filename):
            cube.attributes['filename'] = os.path.basename(filename)
            cube.attributes['lbyr'] = field.lbyr
        fname = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        cube = iris.load_cube(fname, callback=pp_callback)
        self.assertCML(cube, ['uri_callback', 'pp_global.cml'])


if __name__ == "__main__":
    tests.main()
