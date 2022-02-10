# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.io._generate_cubes` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path

import iris


class TestGenerateCubes(tests.IrisTest):
    def test_pathlib_paths(self):
        test_variants = [
            ("string", "string"),
            (["string"], "string"),
            (Path("string"), Path("string")),
        ]

        decode_uri_mock = self.patch(
            "iris.iris.io.decode_uri", return_value=("file", None)
        )
        self.patch("iris.iris.io.load_files")

        for gc_arg, du_arg in test_variants:
            decode_uri_mock.reset_mock()
            list(iris._generate_cubes(gc_arg, None, None))
            decode_uri_mock.assert_called_with(du_arg)


if __name__ == "__main__":
    tests.main()
