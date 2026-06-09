# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for attribute-related loading and saving netcdf files."""

from contextlib import contextmanager

from cf_units import Unit
import pytest

import iris
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION
from iris.tests import _shared_utils


class TestUmVersionAttribute:
    def test_single_saves_as_global(self, tmp_path, request):
        cube = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        nc_path = tmp_path / "test.nc"
        iris.save(cube, nc_path)
        _shared_utils.assert_CDL(request, nc_path)

    def test_multiple_same_saves_as_global(self, tmp_path, request):
        cube_a = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        cube_b = Cube(
            [1.0],
            standard_name="air_pressure",
            units="hPa",
            attributes={"um_version": "4.3"},
        )
        nc_path = tmp_path / "test.nc"
        iris.save(CubeList([cube_a, cube_b]), nc_path)
        _shared_utils.assert_CDL(request, nc_path)

    def test_multiple_different_saves_on_variables(self, tmp_path, request):
        cube_a = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        cube_b = Cube(
            [1.0],
            standard_name="air_pressure",
            units="hPa",
            attributes={"um_version": "4.4"},
        )
        nc_path = tmp_path / "test.nc"
        iris.save(CubeList([cube_a, cube_b]), nc_path)
        _shared_utils.assert_CDL(request, nc_path)


@contextmanager
def _patch_site_configuration(mocker):
    def cf_patch_conventions(conventions):
        return ", ".join([conventions, "convention1, convention2"])

    def update(config):
        config["cf_profile"] = mocker.Mock(name="cf_profile")
        config["cf_patch"] = mocker.Mock(name="cf_patch")
        config["cf_patch_conventions"] = cf_patch_conventions

    orig_site_config = iris.site_configuration.copy()
    update(iris.site_configuration)
    yield
    iris.site_configuration = orig_site_config


class TestConventionsAttributes:
    def test_patching_conventions_attribute(self, tmp_path, mocker):
        # Ensure that user defined conventions are wiped and those which are
        # saved patched through site_config can be loaded without an exception
        # being raised.
        cube = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"Conventions": "some user defined conventions"},
        )

        # Patch the site configuration dictionary.
        nc_path = tmp_path / "test.nc"
        with _patch_site_configuration(mocker):
            iris.save(cube, nc_path)
            res = iris.load_cube(nc_path)

        assert res.attributes["Conventions"] == "{}, {}, {}".format(
            CF_CONVENTIONS_VERSION, "convention1", "convention2"
        )


class TestStandardName:
    def test_standard_name_roundtrip(self, tmp_path):
        standard_name = "air_temperature detection_minimum"
        cube = iris.cube.Cube(1, standard_name=standard_name)
        fout = tmp_path / "standard_name.nc"
        iris.save(cube, fout)
        detection_limit_cube = iris.load_cube(fout)
        assert detection_limit_cube.standard_name == standard_name


class TestCalendar:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.calendar = Unit("days since 1970-01-01", calendar="360_day")
        self.cube = iris.cube.Cube(1, units=self.calendar)

    def test_calendar_roundtrip(self, tmp_path):
        fout = tmp_path / "calendar.nc"
        iris.save(self.cube, fout)
        detection_limit_cube = iris.load_cube(fout)
        assert detection_limit_cube.units == self.calendar
