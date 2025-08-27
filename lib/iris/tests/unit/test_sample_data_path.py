# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.sample_data_path` class."""

import os
import os.path

import pytest

from iris import sample_data_path
from iris.tests import _shared_utils


@_shared_utils.skip_sample_data
class TestIrisSampleData_path:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.sample_dir = tmp_path

    def test_path(self, mocker):
        mocker.patch("iris_sample_data.path", self.sample_dir)
        import iris_sample_data

        assert iris_sample_data.path == self.sample_dir

    def test_call(self, mocker):
        sample_file = self.sample_dir / "sample.txt"
        sample_file.touch()

        mocker.patch("iris_sample_data.path", self.sample_dir)
        result = sample_data_path(os.path.basename(sample_file))
        assert result == str(sample_file)

    def test_file_not_found(self, mocker):
        mocker.patch("iris_sample_data.path", self.sample_dir)
        with pytest.raises(ValueError, match="Sample data .* not found"):
            sample_data_path("foo")

    def test_file_absolute(self, mocker):
        mocker.patch("iris_sample_data.path", self.sample_dir)
        with pytest.raises(ValueError, match="Absolute path"):
            sample_data_path(os.path.abspath("foo"))

    def test_glob_ok(self, mocker):
        sample_path = self.sample_dir / "sample.txt"
        sample_path.touch()

        sample_glob = "?" + os.path.basename(sample_path)[1:]
        mocker.patch("iris_sample_data.path", self.sample_dir)
        result = sample_data_path(sample_glob)
        assert result == os.path.join(self.sample_dir, sample_glob)

    def test_glob_not_found(self, mocker):
        mocker.patch("iris_sample_data.path", self.sample_dir)
        with pytest.raises(ValueError, match="Sample data .* not found"):
            sample_data_path("foo.*")

    def test_glob_absolute(self, mocker):
        mocker.patch("iris_sample_data.path", self.sample_dir)
        with pytest.raises(ValueError, match="Absolute path"):
            sample_data_path(os.path.abspath("foo.*"))


class TestIrisSampleDataMissing:
    def test_no_iris_sample_data(self, mocker):
        mocker.patch("iris.iris_sample_data", None)
        with pytest.raises(ImportError, match="Please install"):
            sample_data_path("")
