# (C) British Crown Copyright 2014 - 2018, Met Office
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
Unit tests for the function
:func:`iris.fileformats.um.um_to_pp`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.fileformats.um import um_to_pp
from iris.tests import mock


class Test_call(tests.IrisTest):
    def test__call(self):
        # Check that the function creates an FF2PP and returns the result
        # of iterating over it.

        # Make a real (test) iterator object, as otherwise iter() complains...
        mock_iterator = (1 for x in ())
        # Make a mock for the iter() call of an FF2PP object.
        mock_iter_call = mock.MagicMock(return_value=mock_iterator)
        # Make a mock FF2PP object instance.
        mock_ff2pp_instance = mock.MagicMock(__iter__=mock_iter_call)
        # Make the mock FF2PP class.
        mock_ff2pp_class = mock.MagicMock(return_value=mock_ff2pp_instance)

        # Call um_to_pp while patching the um._ff_replacement.FF2PP class.
        test_path = '/any/old/file.name'
        with mock.patch('iris.fileformats.um._ff_replacement.FF2PP',
                        mock_ff2pp_class):
            result = um_to_pp(test_path)

        # Check that it called FF2PP in the expected way.
        self.assertEqual(mock_ff2pp_class.call_args_list,
                         [mock.call('/any/old/file.name', read_data=False)])
        self.assertEqual(mock_ff2pp_instance.__iter__.call_args_list,
                         [mock.call()])

        # Check that it returned the expected result.
        self.assertIs(result, mock_iterator)


if __name__ == "__main__":
    tests.main()
