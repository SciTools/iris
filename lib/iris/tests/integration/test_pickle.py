# (C) British Crown Copyright 2014 - 2017, Met Office
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
"""Integration tests for pickling things."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import six.moves.cPickle as pickle

import iris
if tests.GRIB_AVAILABLE:
    import gribapi
    from iris_grib.message import GribMessage


@tests.skip_data
@tests.skip_grib
class TestGribMessage(tests.IrisTest):
    def test(self):
        # Check that a GribMessage pickles without errors.
        path = tests.get_data_path(('GRIB', 'fp_units', 'hours.grib2'))
        messages = GribMessage.messages_from_filename(path)
        message = next(messages)
        with self.temp_filename('.pkl') as filename:
            with open(filename, 'wb') as f:
                pickle.dump(message, f)

    def test_data(self):
        # Check that GribMessage.data pickles without errors.
        path = tests.get_data_path(('GRIB', 'fp_units', 'hours.grib2'))
        messages = GribMessage.messages_from_filename(path)
        message = next(messages)
        with self.temp_filename('.pkl') as filename:
            with open(filename, 'wb') as f:
                pickle.dump(message.data, f)


class Common(object):
    # Ensure that data proxies are pickleable.
    def pickle_cube(self, path):
        cube = iris.load(path)[0]
        with self.temp_filename('.pkl') as filename:
            with open(filename, 'wb') as f:
                pickle.dump(cube, f)


@tests.skip_data
class test_netcdf(Common, tests.IrisTest):
    def test(self):
        path = tests.get_data_path(('NetCDF', 'global', 'xyt',
                                    'SMALL_hires_wind_u_for_ipcc4.nc'))
        self.pickle_cube(path)


@tests.skip_data
class test_pp(Common, tests.IrisTest):
    def test(self):
        path = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        self.pickle_cube(path)


@tests.skip_data
class test_ff(Common, tests.IrisTest):
    def test(self):
        path = tests.get_data_path(('FF', 'n48_multi_field'))
        self.pickle_cube(path)


if __name__ == '__main__':
    tests.main()
