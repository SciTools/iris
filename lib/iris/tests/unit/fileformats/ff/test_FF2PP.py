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
"""Unit tests for the `iris.fileformats.ff.FF2PP` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from copy import deepcopy

import mock

import iris.fileformats.ff as ff


class Test_FF2PP___iter__(tests.IrisTest):
    def test_call_structure(self):
        # Check that the iter method calls the two necessary utility
        # functions
        extract_result = mock.Mock()
        with mock.patch('iris.fileformats.pp._interpret_fields',
                        return_value=iter([])) as interpret:
            with mock.patch('iris.fileformats.ff.FF2PP._extract_field',
                            return_value=extract_result) as extract:
                with mock.patch('iris.fileformats.ff.FFHeader'):
                    list(iter(ff.FF2PP('mock', read_data='read_data_value')))
        interpret.assert_called_once_with(extract_result,
                                          read_data='read_data_value')
        extract.assert_called_once_with()

if __name__ == "__main__":
    tests.main()
