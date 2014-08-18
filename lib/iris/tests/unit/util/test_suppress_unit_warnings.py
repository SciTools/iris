# (C) British Crown Copyright 2014, Met Office
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
"""Test function :func:`iris.util.suppress_unit_warnings`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import warnings

import iris
import iris.tests.stock as stock
from iris.util import suppress_unit_warnings


class Test(tests.IrisTest):

    def setUp(self):
        self.realistic_warning = ("Ignoring netCDF variable 'AL' invalid "
                                  "units '(0 - 1)'")

    def catch_warnings(self, func):
        with warnings.catch_warnings(record=True) as filtered_warnings:
            with suppress_unit_warnings():
                func()
        # Get to the actual warning strings for testing purposes.
        filtered_warnings_list = [w.message.message for w in filtered_warnings]
        return filtered_warnings_list

    def generate_generic_warnings(self):
        # Generate some generic warnings, including one that matches a warning
        # that the function being tested should suppress.
        warnings.warn('Example warning')
        warnings.warn(self.realistic_warning)

    def test_generic_warnings(self):
        filtered = self.catch_warnings(self.generate_generic_warnings)
        self.assertNotIn(self.realistic_warning, filtered)

    def load_captured_warnings_cube(self):
        # Loads data that raises warnings that the function being tested
        # should capture.
        test_filename = tests.get_data_path(('NetCDF', 'testing', 'units.nc'))
        iris.load(test_filename)

    def test_load_file_with_captured_warnings(self):
        filtered = self.catch_warnings(self.load_captured_warnings_cube)
        self.assertEqual(len(filtered), 0)
