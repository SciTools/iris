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
"""Unit tests for :func:`iris.fileformats.rules._make_cube`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.rules import _make_cube


class Test(tests.IrisTest):
    def test_invalid_units(self):
        # Mock converter() function that returns an invalid
        # units string amongst the collection of other elements.
        factories = None
        references = None
        standard_name = None
        long_name = None
        units = 'wibble'  # Invalid unit.
        attributes = dict(source='test')
        cell_methods = None
        dim_coords_and_dims = None
        aux_coords_and_dims = None
        converter = mock.Mock(return_value=(factories,
                                            references,
                                            standard_name,
                                            long_name,
                                            units,
                                            attributes,
                                            cell_methods,
                                            dim_coords_and_dims,
                                            aux_coords_and_dims))

        field = mock.Mock()
        with mock.patch('warnings.warn') as warn:
            cube, factories, references = _make_cube(field, converter)

        # Check attributes dictionary is correctly populated.
        expected_attributes = attributes.copy()
        expected_attributes['invalid_units'] = units
        self.assertEqual(cube.attributes, expected_attributes)

        # Check warning was raised.
        self.assertEqual(warn.call_count, 1)
        warning_msg = warn.call_args[0][0]
        self.assertIn('invalid units', warning_msg)


if __name__ == "__main__":
    tests.main()
