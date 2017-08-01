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
"""
Unit tests for
:func:`iris.fileformats.grib._save_rules.data_section`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

import iris.cube

from iris.fileformats.grib._save_rules import data_section
from iris.tests import mock


GRIB_API = 'iris.fileformats.grib._save_rules.gribapi'
GRIB_MESSAGE = mock.sentinel.GRIB_MESSAGE


class TestMDI(tests.IrisTest):
    def assertBitmapOff(self, grib_api):
        # Check the use of a mask has been turned off via:
        #   gribapi.grib_set(grib_message, 'bitmapPresent', 0)
        grib_api.grib_set.assert_called_once_with(GRIB_MESSAGE,
                                                  'bitmapPresent', 0)

    def assertBitmapOn(self, grib_api, fill_value):
        # Check the use of a mask has been turned on via:
        #   gribapi.grib_set(grib_message, 'bitmapPresent', 1)
        #   gribapi.grib_set_double(grib_message, 'missingValue', fill_value)
        grib_api.grib_set.assert_called_once_with(GRIB_MESSAGE,
                                                  'bitmapPresent', 1)
        grib_api.grib_set_double.assert_called_once_with(GRIB_MESSAGE,
                                                         'missingValue',
                                                         fill_value)

    def assertBitmapRange(self, grib_api, min_data, max_data):
        # Check the use of a mask has been turned on via:
        #   gribapi.grib_set(grib_message, 'bitmapPresent', 1)
        #   gribapi.grib_set_double(grib_message, 'missingValue', ...)
        # and that a suitable fill value has been chosen.
        grib_api.grib_set.assert_called_once_with(GRIB_MESSAGE,
                                                  'bitmapPresent', 1)
        args, = grib_api.grib_set_double.call_args_list
        (message, key, fill_value), kwargs = args
        self.assertIs(message, GRIB_MESSAGE)
        self.assertEqual(key, 'missingValue')
        self.assertTrue(fill_value < min_data or fill_value > max_data,
                        'Fill value {} is not outside data range '
                        '{} to {}.'.format(fill_value, min_data, max_data))
        return fill_value

    def assertValues(self, grib_api, values):
        # Check the correct data values have been set via:
        #   gribapi.grib_set_double_array(grib_message, 'values', ...)
        args, = grib_api.grib_set_double_array.call_args_list
        (message, key, values), kwargs = args
        self.assertIs(message, GRIB_MESSAGE)
        self.assertEqual(key, 'values')
        self.assertArrayEqual(values, values)
        self.assertEqual(kwargs, {})

    def test_simple(self):
        # Check the simple case of non-masked data with no scaling.
        cube = iris.cube.Cube(np.arange(5))
        grib_message = mock.sentinel.GRIB_MESSAGE
        with mock.patch(GRIB_API) as grib_api:
            data_section(cube, grib_message)
        # Check the use of a mask has been turned off.
        self.assertBitmapOff(grib_api)
        # Check the correct data values have been set.
        self.assertValues(grib_api, np.arange(5))

    def test_scaled(self):
        # If the Cube's units don't match the units required by GRIB
        # ensure the data values are scaled correctly.
        cube = iris.cube.Cube(np.arange(5),
                              standard_name='geopotential_height', units='km')
        grib_message = mock.sentinel.GRIB_MESSAGE
        with mock.patch(GRIB_API) as grib_api:
            data_section(cube, grib_message)
        # Check the use of a mask has been turned off.
        self.assertBitmapOff(grib_api)
        # Check the correct data values have been set.
        self.assertValues(grib_api, np.arange(5) * 1000)


if __name__ == "__main__":
    tests.main()
