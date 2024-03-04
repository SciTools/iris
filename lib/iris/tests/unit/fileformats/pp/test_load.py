# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp.load` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import iris.fileformats.pp as pp


class Test_load(tests.IrisTest):
    def test_call_structure(self):
        # Check that the load function calls the two necessary utility
        # functions.
        extract_result = mock.Mock()
        interpret_patch = mock.patch(
            "iris.fileformats.pp._interpret_fields",
            autospec=True,
            return_value=iter([]),
        )
        field_gen_patch = mock.patch(
            "iris.fileformats.pp._field_gen",
            autospec=True,
            return_value=extract_result,
        )
        with interpret_patch as interpret, field_gen_patch as field_gen:
            pp.load("mock", read_data=True)

        interpret.assert_called_once_with(extract_result)
        field_gen.assert_called_once_with(
            "mock", read_data_bytes=True, little_ended=False
        )


if __name__ == "__main__":
    tests.main()
