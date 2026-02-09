# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the function
:func:`iris.fileformats.um.um_to_pp`.

"""

from iris.fileformats.um import um_to_pp


class Test_call:
    def test__call(self, mocker):
        # Check that the function creates an FF2PP and returns the result
        # of iterating over it.

        # Make a real (test) iterator object, as otherwise iter() complains...
        mock_iterator = (1 for x in ())
        # Make a mock for the iter() call of an FF2PP object.
        mock_iter_call = mocker.MagicMock(return_value=mock_iterator)
        # Make a mock FF2PP object instance.
        mock_ff2pp_instance = mocker.MagicMock(__iter__=mock_iter_call)
        # Make the mock FF2PP class.
        mock_ff2pp_class = mocker.MagicMock(return_value=mock_ff2pp_instance)

        # Call um_to_pp while patching the um._ff_replacement.FF2PP class.
        test_path = "/any/old/file.name"
        _ = mocker.patch("iris.fileformats.um._ff_replacement.FF2PP", mock_ff2pp_class)
        result = um_to_pp(test_path)

        # Check that it called FF2PP in the expected way.
        assert mock_ff2pp_class.call_args_list == [
            mocker.call("/any/old/file.name", read_data=False)
        ]
        assert mock_ff2pp_instance.__iter__.call_args_list == [mocker.call()]

        # Check that it returned the expected result.
        assert result is mock_iterator
