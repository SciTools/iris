# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from os.path import dirname
from os.path import join as path_join
from os.path import sep as os_sep
import shutil
from subprocess import check_call
import tempfile

import iris
from iris.tests import stock


class TestClimatology(iris.tests.IrisTest):
    reference_cdl_path = os_sep.join(
        [
            dirname(tests.__file__),
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

    @classmethod
    def setUpClass(cls):
        # Create a temp directory for temp files.
        cls.temp_dir = tempfile.mkdtemp()
        cls.path_ref_cdl = path_join(cls.temp_dir, "standard.cdl")
        cls.path_ref_nc = path_join(cls.temp_dir, "standard.nc")
        # Create reference CDL file.
        with open(cls.path_ref_cdl, "w") as f_out:
            f_out.write(cls._simple_cdl_string())
        # Create reference netCDF file from reference CDL.
        command = "ncgen -o {} {}".format(cls.path_ref_nc, cls.path_ref_cdl)
        check_call(command, shell=True)
        cls.path_temp_nc = path_join(cls.temp_dir, "tmp.nc")

        # Create reference cube.
        cls.cube_ref = stock.climatology_3d()

    @classmethod
    def tearDownClass(cls):
        # Destroy a temp directory for temp files.
        shutil.rmtree(cls.temp_dir)

    ###############################################################################
    # Round-trip tests

    def test_cube_to_cube(self):
        # Save reference cube to file, load cube from same file, test against
        # reference cube.
        iris.save(self.cube_ref, self.path_temp_nc)
        cube = self._load_sanitised_cube(self.path_temp_nc)
        self.assertEqual(cube, self.cube_ref)

    def test_file_to_file(self):
        # Load cube from reference file, save same cube to file, test against
        # reference CDL.
        cube = iris.load_cube(self.path_ref_nc)
        iris.save(cube, self.path_temp_nc)
        self.assertCDL(
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
        self.assertEqual(cube, self.cube_ref)


if __name__ == "__main__":
    tests.main()
