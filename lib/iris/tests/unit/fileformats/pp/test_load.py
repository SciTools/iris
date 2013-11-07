# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris.fileformats.pp.load` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris.fileformats.pp as pp


class Test_load(tests.IrisTest):
    def test_call_structure(self):
        # Check that the load function calls the two necessary utility
        # functions.
        extract_result = mock.Mock()
        interpret_patch = mock.patch('iris.fileformats.pp._interpret_fields',
                                     autospec=True, return_value=iter([]))
        field_gen_patch = mock.patch('iris.fileformats.pp._field_gen',
                                     autospec=True,
                                     return_value=extract_result)
        with interpret_patch as interpret, field_gen_patch as field_gen:
            pp.load('mock', read_data=True)

        interpret.assert_called_once_with(extract_result)
        field_gen.assert_called_once_with('mock', read_data_bytes=True)


if __name__ == "__main__":
    tests.main()
