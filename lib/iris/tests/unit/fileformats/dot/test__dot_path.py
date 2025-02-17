# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.fileformats.dot._dot_path`."""

import os.path
import subprocess

import pytest

from iris.fileformats.dot import _DOT_EXECUTABLE_PATH, _dot_path


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        # Because _dot_path is triggered by the initial import we
        # reset the caching status to allow us to see what happens
        # under different circumstances.
        mocker.patch("iris.fileformats.dot._DOT_CHECKED", False)
        # Also patch the private path variable to the existing value (i.e. no
        # change), and restore it after each test:  As these tests modify it,
        # that can potentially break subsequent 'normal' behaviour.
        mocker.patch("iris.fileformats.dot._DOT_EXECUTABLE_PATH", _DOT_EXECUTABLE_PATH)

    def test_valid_absolute_path(self, mocker):
        # Override the configuration value for System.dot_path
        real_path = os.path.abspath(__file__)
        assert os.path.exists(real_path) and os.path.isabs(real_path)
        mocker.patch("iris.config.get_option", return_value=real_path)
        result = _dot_path()
        assert result == real_path

    def test_invalid_absolute_path(self, mocker):
        # Override the configuration value for System.dot_path
        dummy_path = "/not_a_real_path" * 10
        assert not os.path.exists(dummy_path)
        mocker.patch("iris.config.get_option", return_value=dummy_path)
        result = _dot_path()
        assert result is None

    def test_valid_relative_path(self, mocker):
        # Override the configuration value for System.dot_path
        dummy_path = "not_a_real_path" * 10
        assert not os.path.exists(dummy_path)
        mocker.patch("iris.config.get_option", return_value=dummy_path)
        # Pretend we have a valid installation of dot
        mocker.patch("subprocess.check_output")
        result = _dot_path()
        assert result == dummy_path

    def test_valid_relative_path_broken_install(self, mocker):
        # Override the configuration value for System.dot_path
        dummy_path = "not_a_real_path" * 10
        assert not os.path.exists(dummy_path)
        mocker.patch("iris.config.get_option", return_value=dummy_path)
        # Pretend we have a broken installation of dot
        error = subprocess.CalledProcessError(-5, "foo", "bar")
        mocker.patch("subprocess.check_output", side_effect=error)
        result = _dot_path()
        assert result is None
