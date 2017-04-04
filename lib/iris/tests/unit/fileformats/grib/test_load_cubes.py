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
"""Unit tests for the `iris.fileformats.grib.load_cubes` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import iris.tests as tests

import mock

import iris
from iris.fileformats.rules import Loader

import iris.fileformats.grib
from iris.fileformats.grib import load_cubes


class Test(tests.IrisGribTest):
    def test(self):
        generator = iris.fileformats.grib._load_generate
        converter = iris.fileformats.grib._load_convert.convert
        files = mock.sentinel.FILES
        callback = mock.sentinel.CALLBACK
        expected_result = mock.sentinel.RESULT
        with mock.patch('iris.fileformats.rules.load_cubes') as rules_load:
            rules_load.return_value = expected_result
            result = load_cubes(files, callback)
            kwargs = {}
            loader = Loader(generator, kwargs, converter, None)
            rules_load.assert_called_once_with(files, callback, loader)
            self.assertIs(result, expected_result)


@tests.skip_data
class Test_load_cubes(tests.IrisGribTest):

    def test_reduced_raw(self):
        # Loading a GRIB message defined on a reduced grid without
        # interpolating to a regular grid.
        gribfile = tests.get_data_path(
            ("GRIB", "reduced", "reduced_gg.grib2"))
        grib_generator = load_cubes(gribfile)
        self.assertCML(next(grib_generator))


if __name__ == "__main__":
    tests.main()
