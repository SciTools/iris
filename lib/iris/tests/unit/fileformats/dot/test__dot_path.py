# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :func:`iris.fileformats.dot._dot_path`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import os.path
import subprocess
from unittest import mock

from iris.fileformats.dot import _DOT_EXECUTABLE_PATH, _dot_path


class Test(tests.IrisTest):
    def setUp(self):
        # Because _dot_path is triggered by the initial import we
        # reset the caching status to allow us to see what happens
        # under different circumstances.
        self.patch("iris.fileformats.dot._DOT_CHECKED", new=False)
        # Also patch the private path variable to the existing value (i.e. no
        # change), and restore it after each test:  As these tests modify it,
        # that can potentially break subsequent 'normal' behaviour.
        self.patch(
            "iris.fileformats.dot._DOT_EXECUTABLE_PATH", _DOT_EXECUTABLE_PATH
        )

    def test_valid_absolute_path(self):
        # Override the configuration value for System.dot_path
        real_path = os.path.abspath(__file__)
        assert os.path.exists(real_path) and os.path.isabs(real_path)
        with mock.patch("iris.config.get_option", return_value=real_path):
            result = _dot_path()
        self.assertEqual(result, real_path)

    def test_invalid_absolute_path(self):
        # Override the configuration value for System.dot_path
        dummy_path = "/not_a_real_path" * 10
        assert not os.path.exists(dummy_path)
        with mock.patch("iris.config.get_option", return_value=dummy_path):
            result = _dot_path()
        self.assertIsNone(result)

    def test_valid_relative_path(self):
        # Override the configuration value for System.dot_path
        dummy_path = "not_a_real_path" * 10
        assert not os.path.exists(dummy_path)
        with mock.patch("iris.config.get_option", return_value=dummy_path):
            # Pretend we have a valid installation of dot
            with mock.patch("subprocess.check_output"):
                result = _dot_path()
        self.assertEqual(result, dummy_path)

    def test_valid_relative_path_broken_install(self):
        # Override the configuration value for System.dot_path
        dummy_path = "not_a_real_path" * 10
        assert not os.path.exists(dummy_path)
        with mock.patch("iris.config.get_option", return_value=dummy_path):
            # Pretend we have a broken installation of dot
            error = subprocess.CalledProcessError(-5, "foo", "bar")
            with mock.patch("subprocess.check_output", side_effect=error):
                result = _dot_path()
        self.assertIsNone(result)


if __name__ == "__main__":
    tests.main()
