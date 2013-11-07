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

import mock

import iris.fileformats.ff as ff


class Test_FF2PP___iter__(tests.IrisTest):
    @mock.patch('iris.fileformats.ff.FFHeader')
    def test_call_structure(self, _FFHeader):
        # Check that the iter method calls the two necessary utility
        # functions
        extract_result = mock.Mock()
        interpret_patch = mock.patch('iris.fileformats.pp._interpret_fields',
                                     autospec=True, return_value=iter([]))
        extract_patch = mock.patch('iris.fileformats.ff.FF2PP._extract_field',
                                   autospec=True, return_value=extract_result)

        FF2PP_instance = ff.FF2PP('mock')
        with interpret_patch as interpret, extract_patch as extract:
            list(iter(FF2PP_instance))

        interpret.assert_called_once_with(extract_result)
        extract.assert_called_once_with(FF2PP_instance)


if __name__ == "__main__":
    tests.main()
