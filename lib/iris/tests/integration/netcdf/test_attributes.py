# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for attribute-related loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from contextlib import contextmanager
from unittest import mock

import iris
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION


class TestUmVersionAttribute(tests.IrisTest):
    def test_single_saves_as_global(self):
        cube = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        with self.temp_filename(".nc") as nc_path:
            iris.save(cube, nc_path)
            self.assertCDL(nc_path)

    def test_multiple_same_saves_as_global(self):
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
        with self.temp_filename(".nc") as nc_path:
            iris.save(CubeList([cube_a, cube_b]), nc_path)
            self.assertCDL(nc_path)

    def test_multiple_different_saves_on_variables(self):
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
        with self.temp_filename(".nc") as nc_path:
            iris.save(CubeList([cube_a, cube_b]), nc_path)
            self.assertCDL(nc_path)


@contextmanager
def _patch_site_configuration():
    def cf_patch_conventions(conventions):
        return ", ".join([conventions, "convention1, convention2"])

    def update(config):
        config["cf_profile"] = mock.Mock(name="cf_profile")
        config["cf_patch"] = mock.Mock(name="cf_patch")
        config["cf_patch_conventions"] = cf_patch_conventions

    orig_site_config = iris.site_configuration.copy()
    update(iris.site_configuration)
    yield
    iris.site_configuration = orig_site_config


class TestConventionsAttributes(tests.IrisTest):
    def test_patching_conventions_attribute(self):
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
        with _patch_site_configuration(), self.temp_filename(".nc") as nc_path:
            iris.save(cube, nc_path)
            res = iris.load_cube(nc_path)

        self.assertEqual(
            res.attributes["Conventions"],
            "{}, {}, {}".format(
                CF_CONVENTIONS_VERSION, "convention1", "convention2"
            ),
        )


class TestStandardName(tests.IrisTest):
    def test_standard_name_roundtrip(self):
        standard_name = "air_temperature detection_minimum"
        cube = iris.cube.Cube(1, standard_name=standard_name)
        with self.temp_filename(suffix=".nc") as fout:
            iris.save(cube, fout)
            detection_limit_cube = iris.load_cube(fout)
            self.assertEqual(detection_limit_cube.standard_name, standard_name)


if __name__ == "__main__":
    tests.main()
