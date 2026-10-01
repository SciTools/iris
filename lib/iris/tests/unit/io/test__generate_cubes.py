# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.io._generate_cubes` function."""

from pathlib import Path

import pytest

from iris.loading import _generate_cubes


class TestGenerateCubes:
    def test_pathlib_paths(self, mocker):
        test_variants = [
            ("string", "string"),
            (["string"], "string"),
            (Path("string"), Path("string")),
        ]

        decode_uri_mock = mocker.patch(
            "iris.iris.io.decode_uri", return_value=("file", None, None)
        )
        mocker.patch("iris.iris.io.load_files")

        for gc_arg, du_arg in test_variants:
            decode_uri_mock.reset_mock()
            list(_generate_cubes(gc_arg, None, None))
            decode_uri_mock.assert_called_with(du_arg)

    @pytest.mark.parametrize(
        "uri",
        [
            "file:///foo#mode=nczarr",
            "file:///foo#mode=xarray",
            "https://foo#mode=nczarr",
        ],
    )
    def test_nczarr_uri(self, uri, mocker):
        load_files_mock = mocker.patch("iris.iris.io.load_files", return_value=[])
        load_http_mock = mocker.patch("iris.iris.io.load_http", return_value=[])

        list(_generate_cubes(uri, None, None))

        load_http_mock.assert_called_once_with([uri], None)
        load_files_mock.assert_not_called()

    def test_non_nczarr_uri(self, mocker):
        uri = "file:///foo#bar=baz"
        load_files_mock = mocker.patch("iris.iris.io.load_files", return_value=[])
        load_http_mock = mocker.patch("iris.iris.io.load_http", return_value=[])

        list(_generate_cubes(uri, None, None))

        load_files_mock.assert_called_once_with(["///foo#bar=baz"], None, None)
        load_http_mock.assert_not_called()
