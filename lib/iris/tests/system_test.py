# (C) British Crown Copyright 2010 - 2015, Met Office
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
This system test module is useful to identify if some of the key components required for Iris are available.

The system tests can be run with ``python setup.py test --system-tests``.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before importing anything else

import numpy as np

import iris
import iris.fileformats.grib as grib
import iris.fileformats.netcdf as netcdf
import iris.fileformats.pp as pp
import iris.tests as tests
import iris.unit

class SystemInitialTest(tests.IrisTest):

    def system_test_supported_filetypes(self):
        nx, ny = 60, 60
        dataarray = np.arange(nx * ny, dtype='>f4').reshape(nx, ny)

        laty = np.linspace(0, 59, ny).astype('f8')
        lonx = np.linspace(30, 89, nx).astype('f8')

        horiz_cs = lambda : iris.coord_systems.GeogCS(6371229)

        cm = iris.cube.Cube(data=dataarray, long_name="System test data", units='m s-1')
        cm.add_dim_coord(
            iris.coords.DimCoord(laty, 'latitude', units='degrees',
                                 coord_system=horiz_cs()),
            0)
        cm.add_dim_coord(
            iris.coords.DimCoord(lonx, 'longitude', units='degrees',
                coord_system=horiz_cs()),
            1)
        cm.add_aux_coord(iris.coords.AuxCoord(np.array([9], 'i8'),
                                              'forecast_period', units='hours'))
        hours_since_epoch = iris.unit.Unit('hours since epoch',
                                           iris.unit.CALENDAR_GREGORIAN)
        cm.add_aux_coord(iris.coords.AuxCoord(np.array([3], 'i8'),
                                              'time', units=hours_since_epoch))
        cm.add_aux_coord(iris.coords.AuxCoord(np.array([99], 'i8'),
                                              long_name='pressure', units='Pa'))

        cm.assert_valid()

        for filetype in ('.nc', '.pp', '.grib2'):
            saved_tmpfile = iris.util.create_temp_filename(suffix=filetype)
            iris.save(cm, saved_tmpfile)

            new_cube = iris.load_cube(saved_tmpfile)

            self.assertCML(new_cube, ('system', 'supported_filetype_%s.cml' % filetype))

    def system_test_grib_patch(self):
        import gribapi
        gm = gribapi.grib_new_from_samples("GRIB2")
        result = gribapi.grib_get_double(gm, "missingValue")

        new_missing_value = 123456.0
        gribapi.grib_set_double(gm, "missingValue", new_missing_value)
        new_result = gribapi.grib_get_double(gm, "missingValue")

        self.assertEqual(new_result, new_missing_value)

    def system_test_imports_general(self):
        if tests.MPL_AVAILABLE:
            import matplotlib
        import netCDF4


if __name__ == '__main__':
    tests.main()
