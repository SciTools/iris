# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp.load` function."""

import iris.fileformats.pp as pp


class Test_load:
    def test_call_structure(self, mocker):
        # Check that the load function calls the two necessary utility
        # functions.
        extract_result = mocker.Mock()
        interpret_patch = mocker.patch(
            "iris.fileformats.pp._interpret_fields",
            autospec=True,
            return_value=iter([]),
        )
        field_gen_patch = mocker.patch(
            "iris.fileformats.pp._field_gen",
            autospec=True,
            return_value=extract_result,
        )
        pp.load("mock", read_data=True)

        interpret_patch.assert_called_once_with(extract_result)
        field_gen_patch.assert_called_once_with(
            "mock", read_data_bytes=True, little_ended=False
        )
