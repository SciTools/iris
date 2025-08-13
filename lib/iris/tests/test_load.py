# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the main loading API."""

import pathlib

import pytest

import iris
from iris.fileformats.netcdf import _thread_safe_nc
import iris.io
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestLoad:
    def test_normal(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load(paths)
        assert len(cubes) == 1

    def test_path_object(self):
        paths = (
            pathlib.Path(_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        cubes = iris.load(paths)
        assert len(cubes) == 1

    def test_nonexist(self):
        paths = (
            _shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),
            _shared_utils.get_data_path(["PP", "_guaranteed_non_exist.pp"]),
        )
        with pytest.raises(IOError, match="files specified did not exist"):
            _ = iris.load(paths)

    def test_nonexist_wild(self):
        paths = (
            _shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),
            _shared_utils.get_data_path(["PP", "_guaranteed_non_exist_*.pp"]),
        )
        with pytest.raises(IOError, match="files specified did not exist"):
            _ = iris.load(paths)

    def test_bogus(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load(paths, "wibble")
        assert len(cubes) == 0

    def test_real_and_bogus(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load(paths, ("air_temperature", "wibble"))
        assert len(cubes) == 1

    def test_duplicate(self):
        paths = (
            _shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),
            _shared_utils.get_data_path(["PP", "aPPglob1", "gl?bal.pp"]),
        )
        cubes = iris.load(paths)
        assert len(cubes) == 2


@_shared_utils.skip_data
class TestLoadCube:
    def test_normal(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        _ = iris.load_cube(paths)

    def test_path_object(self):
        paths = (
            pathlib.Path(_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        _ = iris.load_cube(paths)

    def test_not_enough(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        with pytest.raises(iris.exceptions.ConstraintMismatchError):
            iris.load_cube(paths, "wibble")

    def test_too_many(self):
        paths = (
            _shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),
            _shared_utils.get_data_path(["PP", "aPPglob1", "gl?bal.pp"]),
        )
        with pytest.raises(iris.exceptions.ConstraintMismatchError):
            iris.load_cube(paths)


@_shared_utils.skip_data
class TestLoadCubes:
    def test_normal(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load_cubes(paths)
        assert len(cubes) == 1

    def test_path_object(self):
        paths = (
            pathlib.Path(_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        cubes = iris.load_cubes(paths)
        assert len(cubes) == 1

    def test_not_enough(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        with pytest.raises(iris.exceptions.ConstraintMismatchError):
            iris.load_cubes(paths, "wibble")

    def test_not_enough_multi(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        with pytest.raises(iris.exceptions.ConstraintMismatchError):
            iris.load_cubes(paths, ("air_temperature", "wibble"))

    def test_too_many(self):
        paths = (
            _shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),
            _shared_utils.get_data_path(["PP", "aPPglob1", "gl?bal.pp"]),
        )
        with pytest.raises(iris.exceptions.ConstraintMismatchError):
            iris.load_cube(paths)


@_shared_utils.skip_data
class TestLoadRaw:
    def test_normal(self):
        paths = (_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load_raw(paths)
        assert len(cubes) == 1

    def test_path_object(self):
        paths = (
            pathlib.Path(_shared_utils.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        cubes = iris.load_raw(paths)
        assert len(cubes) == 1


class TestOPeNDAP:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.url = "https://geoport.whoi.edu:80/thredds/dodsC/bathy/gom15"

    def test_load_http_called(self):
        # Check that calling iris.load_* with an http URI triggers a call to
        # ``iris.io.load_http``

        class LoadHTTPCalled(Exception):
            pass

        def new_load_http(passed_urls, *args, **kwargs):
            assert len(passed_urls) == 1
            assert self.url == passed_urls[0]
            raise LoadHTTPCalled()

        try:
            orig = iris.io.load_http
            iris.io.load_http = new_load_http

            for fn in [
                iris.load,
                iris.load_raw,
                iris.load_cube,
                iris.load_cubes,
            ]:
                with pytest.raises(LoadHTTPCalled):
                    fn(self.url)

        finally:
            iris.io.load_http = orig

    @_shared_utils.skip_data
    def test_net_cdf_dataset_call(self, mocker):
        # Check that load_http calls netCDF4.Dataset and supplies the expected URL.

        # To avoid making a request to an OPeNDAP server in a test, instead
        # mock the call to netCDF.Dataset so that it returns a dataset for a
        # local file.
        filename = _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
        )
        fake_dataset = _thread_safe_nc.DatasetWrapper(filename)

        dataset_loader = mocker.patch(
            "iris.fileformats.netcdf._thread_safe_nc.DatasetWrapper",
            return_value=fake_dataset,
        )
        next(iris.io.load_http([self.url], callback=None))
        dataset_loader.assert_called_with(self.url, mode="r")
