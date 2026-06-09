# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the file loading mechanism."""

import pytest

import iris
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestFileLoad:
    def _test_file(self, request: pytest.FixtureRequest, src_path, reference_filename):
        """Checks the result of loading the given file spec, or creates the
        reference file if it doesn't exist.

        """
        cubes = iris.load_raw(_shared_utils.get_data_path(src_path))
        _shared_utils.assert_CML(request, cubes, ["file_load", reference_filename])

    def test_no_file(self):
        # Test an IOError is received when a filename is given which doesn't match any files
        real_file = ["PP", "globClim1", "theta.pp"]
        non_existant_file = ["PP", "globClim1", "no_such_file*"]

        with pytest.raises(IOError, match="files specified did not exist"):
            iris.load(_shared_utils.get_data_path(non_existant_file))
        with pytest.raises(IOError, match="files specified did not exist"):
            iris.load(
                [
                    _shared_utils.get_data_path(non_existant_file),
                    _shared_utils.get_data_path(real_file),
                ]
            )
        with pytest.raises(IOError, match="files specified did not exist"):
            iris.load(
                [
                    _shared_utils.get_data_path(real_file),
                    _shared_utils.get_data_path(non_existant_file),
                ]
            )

    def test_single_file(self, request):
        src_path = ["PP", "globClim1", "theta.pp"]
        self._test_file(request, src_path, "theta_levels.cml")

    def test_star_wildcard(self, request):
        src_path = ["PP", "globClim1", "*_wind.pp"]
        self._test_file(request, src_path, "wind_levels.cml")

    def test_query_wildcard(self, request):
        src_path = ["PP", "globClim1", "?_wind.pp"]
        self._test_file(request, src_path, "wind_levels.cml")

    def test_charset_wildcard(self, request):
        src_path = ["PP", "globClim1", "[rstu]_wind.pp"]
        self._test_file(request, src_path, "u_wind_levels.cml")

    def test_negative_charset_wildcard(self, request):
        src_path = ["PP", "globClim1", "[!rstu]_wind.pp"]
        self._test_file(request, src_path, "v_wind_levels.cml")

    def test_empty_file(self, tmp_path):
        temp_filename = tmp_path / "tmp.pp"
        with temp_filename.open("a"):
            with pytest.raises(iris.exceptions.TranslationError):
                iris.load(temp_filename)
