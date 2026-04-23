# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for loading and saving netcdf files."""

from os.path import dirname
from os.path import sep as os_sep

import pytest

import iris
from iris.tests import _shared_utils, stock
from iris.tests.stock.netcdf import ncgen_from_cdl


class TestClimatology:
    reference_cdl_path = os_sep.join(
        [
            dirname(iris.tests.__file__),
            (
                "results/integration/climatology/TestClimatology/"
                "reference_simpledata.cdl"
            ),
        ]
    )

    @classmethod
    def _simple_cdl_string(cls):
        with open(cls.reference_cdl_path, "r") as f:
            cdl_content = f.read()
        # Add the expected CDL first line since this is removed from the
        # stored results file.
        cdl_content = "netcdf {\n" + cdl_content

        return cdl_content

    @staticmethod
    def _load_sanitised_cube(filepath):
        cube = iris.load_cube(filepath)
        # Remove attributes convention, if any.
        cube.attributes.pop("Conventions", None)
        # Remove any var-names.
        for coord in cube.coords():
            coord.var_name = None
        cube.var_name = None
        return cube

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, request, tmp_path_factory):
        # Create a temp directory for temp files.
        cls = request.cls
        cls.temp_dir = tmp_path_factory.mktemp("temp")
        cls.path_ref_cdl = cls.temp_dir / "standard.cdl"
        cls.path_ref_nc = cls.temp_dir / "standard.nc"
        # Create reference CDL and netcdf files (with ncgen).
        ncgen_from_cdl(
            cdl_str=cls._simple_cdl_string(),
            cdl_path=cls.path_ref_cdl,
            nc_path=cls.path_ref_nc,
        )

        cls.path_temp_nc = cls.temp_dir / "tmp.nc"

        # Create reference cube.
        cls.cube_ref = stock.climatology_3d()

    ###############################################################################
    # Round-trip tests

    def test_cube_to_cube(self):
        # Save reference cube to file, load cube from same file, test against
        # reference cube.
        iris.save(self.cube_ref, self.path_temp_nc)
        cube = self._load_sanitised_cube(self.path_temp_nc)
        assert cube == self.cube_ref

    def test_file_to_file(self, request):
        # Load cube from reference file, save same cube to file, test against
        # reference CDL.
        cube = iris.load_cube(self.path_ref_nc)
        iris.save(cube, self.path_temp_nc)
        _shared_utils.assert_CDL(
            request,
            self.path_temp_nc,
            reference_filename=self.reference_cdl_path,
            flags="",
        )

    # NOTE:
    # The saving half of the round-trip tests is tested in the
    # appropriate dedicated test class:
    # unit.fileformats.netcdf.test_Saver.Test_write.test_with_climatology .
    # The loading half has no equivalent dedicated location, so is tested
    # here as test_load_from_file.

    def test_load_from_file(self):
        # Create cube from file, test against reference cube.
        cube = self._load_sanitised_cube(self.path_ref_nc)
        assert cube == self.cube_ref
