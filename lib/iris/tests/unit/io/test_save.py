# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.io.save` function."""

from pathlib import Path

import iris
from iris.cube import Cube


class TestSave:
    def test_pathlib_save(self, mocker):
        file_mock = mocker.Mock()
        # Have to configure after creation because "name" is special
        file_mock.configure_mock(name="string")

        find_saver_mock = mocker.patch(
            "iris.io.find_saver", return_value=(lambda *args, **kwargs: None)
        )

        def replace_expand(file_specs, files_expected=True):
            return file_specs

        # does not expand filepaths due to patch
        mocker.patch("iris.io.expand_filespecs", replace_expand)

        test_variants = [
            ("string", "string"),
            (Path("string/string"), "string/string"),
            (file_mock, "string"),
        ]

        for target, fs_val in test_variants:
            try:
                iris.save(Cube([]), target)
            except ValueError:
                print("ValueError")
                pass
            find_saver_mock.assert_called_with(fs_val)
