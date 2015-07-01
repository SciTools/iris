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
"""Test function :func:`iris.unit.suppress_unit_warnings`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import warnings

import iris
from iris.unit import suppress_unit_warnings


class Test(tests.IrisTest):

    def test_generic_warnings(self):
        unit_warning = ("Ignoring netCDF variable 'AL' invalid "
                        "units '(0 - 1)'")
        example_warning = 'Example warning'
        with warnings.catch_warnings(record=True) as filtered_warnings:
            with suppress_unit_warnings():
                warnings.warn(unit_warning)
                warnings.warn(example_warning)
        # Get to the actual warning strings for testing purposes.
        filtered_warnings_list = [w.message.message for w in filtered_warnings]
        self.assertNotIn(unit_warning, filtered_warnings_list)
        self.assertIn(example_warning, filtered_warnings_list)

    @tests.skip_data
    def test_load_file_with_captured_warnings(self):
        test_filename = tests.get_data_path(('NetCDF', 'testing', 'units.nc'))
        with warnings.catch_warnings(record=True) as filtered_warnings:
            with suppress_unit_warnings():
                iris.load(test_filename)
        filtered_warnings_list = [w.message.message for w in filtered_warnings]
        self.assertEqual(len(filtered_warnings_list), 0)


if __name__ == '__main__':
    tests.main()
