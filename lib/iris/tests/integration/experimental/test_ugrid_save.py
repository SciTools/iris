# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests for NetCDF-UGRID file saving.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
import shutil
import tempfile

import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import iris.fileformats.netcdf
from iris.tests import IrisTest
from iris.tests.stock.mesh import (  # sample_mesh,; sample_meshcoord,
    sample_mesh_cube,
)


class TestBasicSave(IrisTest):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_parts(self):
        cube = sample_mesh_cube()
        # Make this REALLY minimal
        # Note: removing the 'extras' in the 'sample+mesh_cube'
        cube = cube[0]  # Strip out first dimension.
        # # Remove un-needed extra coords (for now).
        for name in ("level", "i_mesh_face", "mesh_face_aux"):
            cube.remove_coord(name)
        print(cube)

        temp_path = str(self.temp_dir / "tmp.nc")
        with iris.fileformats.netcdf.Saver(temp_path, "NETCDF4") as saver:
            saver.write(cube)
        # TODO: do some actual testing, beyond "does not fail"
        # self.assertCDL(temp_path)  # TODO: for now onl

    def test_basic_save(self):
        # Generate an ultra-simple unstructured cube
        cube = sample_mesh_cube()
        print(cube)
        return

        # Save it out, and check that the CDL is as expected.
        tempfile_path = str(self.temp_dir / "basic.nc")
        iris.save(cube, tempfile_path)
        self.assertCDL(tempfile_path)

    # def test_complex_multiple_save(self):
    #     source_dirpath = Path(iris.tests.get_data_path(
    #             ['NetCDF', 'unstructured_grid', 'lfric_surface_mean.nc']))
    #
    #     # Save it out, and check that the CDL is as expected.
    #     tempfile_path = str(self.temp_dir / 'basic.nc')
    #     iris.save(cube, tempfile_path)
    #     self.assertCDL(tempfile_path)

    def test_roundtrip(self):
        source_dirpath = Path(
            iris.tests.get_data_path(["NetCDF", "unstructured_grid"])
        )
        file_name = "data_C4.nc"
        source_filepath = str(source_dirpath / file_name)

        # # Ensure that our test reference text matches the original input file
        # self.assertCDL(source_filepath)

        # Load the data (one cube) and save it out to a temporary file of the same name
        with PARSE_UGRID_ON_LOAD.context():
            cube = iris.load_cube(source_filepath)

        print(cube)
        target_filepath = str(self.temp_dir / file_name)
        iris.save(cube, target_filepath)

        from subprocess import check_output

        print("")
        print("ORIGINAL:")
        text = check_output("ncdump -h " + source_filepath, shell=True)
        print(text.decode())
        print("")
        print("RE-SAVED:")
        text = check_output("ncdump -h " + target_filepath, shell=True)
        print(text.decode())
        # # Ensure that the saved result is identical
        # self.assertCDL(target_filepath)

        # Now try loading BACK...
        with PARSE_UGRID_ON_LOAD.context():
            cube_reload = iris.load_cube(target_filepath)
        print("")
        print("OUTPUT-RE-LOADED:")
        print(cube_reload)

        mesh_orig = cube.mesh
        mesh_reload = cube_reload.mesh
        for propname in dir(mesh_orig):
            if not propname.startswith("_"):
                prop_orig = getattr(mesh_orig, propname)
                if not callable(prop_orig):
                    prop_reload = getattr(mesh_reload, propname)
                    self.assertEqual(
                        (propname, prop_reload), (propname, prop_orig)
                    )


if __name__ == "__main__":
    tests.main()
