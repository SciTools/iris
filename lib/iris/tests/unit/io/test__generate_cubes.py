# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.io._generate_cubes` function."""

from pathlib import Path

from iris.loading import _generate_cubes


class TestGenerateCubes:
    def test_pathlib_paths(self, mocker):
        test_variants = [
            ("string", "string"),
            (["string"], "string"),
            (Path("string"), Path("string")),
        ]

        decode_uri_mock = mocker.patch(
            "iris.iris.io.decode_uri", return_value=("file", None)
        )
        mocker.patch("iris.iris.io.load_files")

        for gc_arg, du_arg in test_variants:
            decode_uri_mock.reset_mock()
            list(_generate_cubes(gc_arg, None, None))
            decode_uri_mock.assert_called_with(du_arg)
