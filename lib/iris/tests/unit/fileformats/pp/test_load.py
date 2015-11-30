# (C) British Crown Copyright 2013 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.fileformats.pp as pp
from iris.tests import mock


class Test_load(tests.IrisTest):
    def setUp(self):
        side_effect = [iter([mock.sentinel.ppfields1]),
                       iter([mock.sentinel.ppfields2,
                             mock.sentinel.ppfields3])]
        patch = mock.patch('iris.fileformats.pp._interpret_fields',
                           autospec=True, side_effect=side_effect)
        self.interpret_patch = patch.start()
        self.addCleanup(patch.stop)

        self.extract_result = mock.Mock()

        side_effect = [[mock.sentinel.hf_ppfields1],
                       [mock.sentinel.hf_ppfields2,
                        mock.sentinel.hf_ppfields3]]
        patch = mock.patch('iris.fileformats.pp._field_gen', autospec=True,
                           side_effect=side_effect)
        self.field_gen_patch = patch.start()
        self.addCleanup(patch.stop)

    def test_call_structure(self):
        # Check that the load function calls the two necessary utility
        # functions.
        pp.load('mock', read_data=True)

        self.interpret_patch.assert_called_once_with(
            [mock.sentinel.hf_ppfields1])
        self.field_gen_patch.assert_called_once_with('mock',
                                                     read_data_bytes=True)

    def test_multi_filename(self):
        # Ensure multi filename support.
        fields = pp.load(['fnme1', 'fnme2'], read_data=True)

        self.assertEqual(self.interpret_patch.call_args_list,
                         [mock.call([mock.sentinel.hf_ppfields1]),
                          mock.call([mock.sentinel.hf_ppfields2,
                                    mock.sentinel.hf_ppfields3])])
        self.assertEqual(self.field_gen_patch.call_args_list,
                         [mock.call('fnme1', read_data_bytes=True),
                          mock.call('fnme2', read_data_bytes=True)])
        self.assertEqual(list(fields),
                         [mock.sentinel.ppfields1, mock.sentinel.ppfields2,
                          mock.sentinel.ppfields3])


if __name__ == "__main__":
    tests.main()
