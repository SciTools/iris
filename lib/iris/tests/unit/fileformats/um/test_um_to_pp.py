# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the function
:func:`iris.fileformats.um.um_to_pp`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.fileformats.um import um_to_pp


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
        test_path = "/any/old/file.name"
        with mock.patch(
            "iris.fileformats.um._ff_replacement.FF2PP", mock_ff2pp_class
        ):
            result = um_to_pp(test_path)

        # Check that it called FF2PP in the expected way.
        self.assertEqual(
            mock_ff2pp_class.call_args_list,
            [mock.call("/any/old/file.name", read_data=False)],
        )
        self.assertEqual(
            mock_ff2pp_instance.__iter__.call_args_list, [mock.call()]
        )

        # Check that it returned the expected result.
        self.assertIs(result, mock_iterator)


if __name__ == "__main__":
    tests.main()
