# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.io.save` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
from unittest import mock

import iris
from iris.cube import Cube


class TestSave(tests.IrisTest):
    def test_pathlib_save(self):
        file_mock = mock.Mock()
        # Have to configure after creation because "name" is special
        file_mock.configure_mock(name="string")

        find_saver_mock = self.patch(
            "iris.io.find_saver", return_value=(lambda *args, **kwargs: None)
        )

        def replace_expand(file_specs, files_expected=True):
            return file_specs

        # does not expand filepaths due to patch
        self.patch("iris.io.expand_filespecs", replace_expand)

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


if __name__ == "__main__":
    tests.main()
