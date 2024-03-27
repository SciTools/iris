# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for NetCDF-UGRID file saving."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import glob
from pathlib import Path
import shutil
import tempfile

import iris
from iris.experimental.ugrid.load import PARSE_UGRID_ON_LOAD
import iris.fileformats.netcdf
from iris.tests.stock.netcdf import _add_standard_data, ncgen_from_cdl


class TestBasicSave(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.examples_dir = (
            Path(__file__).absolute().parent / "ugrid_conventions_examples"
        )
        example_paths = glob.glob(str(cls.examples_dir / "*ex*.cdl"))
        example_names = [
            str(Path(filepath).name).split("_")[1]  # = "ex<N>"
            for filepath in example_paths
        ]
        cls.example_names_paths = {
            name: path for name, path in zip(example_names, example_paths)
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_example_result_cdls(self):
        # Snapshot the result of saving the example cases.
        for ex_name, cdl_path in self.example_names_paths.items():
            # Create a test netcdf file.
            target_ncfile_path = str(self.temp_dir / f"{ex_name}.nc")
            ncgen_from_cdl(cdl_str=None, cdl_path=cdl_path, nc_path=target_ncfile_path)
            # Fill in blank data-variables.
            _add_standard_data(target_ncfile_path)
            # Load as Iris data
            with PARSE_UGRID_ON_LOAD.context():
                cubes = iris.load(target_ncfile_path)
            # Re-save, to check the save behaviour.
            resave_ncfile_path = str(self.temp_dir / f"{ex_name}_resaved.nc")
            iris.save(cubes, resave_ncfile_path)
            # Check the output against a CDL snapshot.
            refdir_relpath = "integration/experimental/ugrid_save/TestBasicSave/"
            reffile_name = str(Path(cdl_path).name).replace(".nc", ".cdl")
            reffile_path = refdir_relpath + reffile_name
            self.assertCDL(resave_ncfile_path, reference_filename=reffile_path)

    def test_example_roundtrips(self):
        # Check that save-and-loadback leaves Iris data unchanged,
        # for data derived from each UGRID example CDL.
        for ex_name, cdl_path in self.example_names_paths.items():
            # Create a test netcdf file.
            target_ncfile_path = str(self.temp_dir / f"{ex_name}.nc")
            ncgen_from_cdl(cdl_str=None, cdl_path=cdl_path, nc_path=target_ncfile_path)
            # Fill in blank data-variables.
            _add_standard_data(target_ncfile_path)
            # Load the original as Iris data
            with PARSE_UGRID_ON_LOAD.context():
                orig_cubes = iris.load(target_ncfile_path)

            if "ex4" in ex_name:
                # Discard the extra formula terms component cubes
                # Saving these does not do what you expect
                orig_cubes = orig_cubes.extract("datavar")

            # Save-and-load-back to compare the Iris saved result.
            resave_ncfile_path = str(self.temp_dir / f"{ex_name}_resaved.nc")
            iris.save(orig_cubes, resave_ncfile_path)
            with PARSE_UGRID_ON_LOAD.context():
                savedloaded_cubes = iris.load(resave_ncfile_path)

            # This should match the original exactly
            # ..EXCEPT for our inability to compare meshes.
            for orig, reloaded in zip(orig_cubes, savedloaded_cubes):
                for cube in (orig, reloaded):
                    # Remove conventions attributes, which may differ.
                    cube.attributes.pop("Conventions", None)
                    # Remove var-names, which may differ.
                    cube.var_name = None

                # Compare the mesh contents (as we can't compare actual meshes)
                self.assertEqual(orig.location, reloaded.location)
                orig_mesh = orig.mesh
                reloaded_mesh = reloaded.mesh
                self.assertEqual(orig_mesh.all_coords, reloaded_mesh.all_coords)
                self.assertEqual(
                    orig_mesh.all_connectivities,
                    reloaded_mesh.all_connectivities,
                )
                # Index the cubes to replace meshes with meshcoord-derived aux coords.
                # This needs [:0] on the mesh dim, so do that on all dims.
                keys = tuple([slice(0, None)] * orig.ndim)
                orig = orig[keys]
                reloaded = reloaded[keys]
                # Resulting cubes, with collapsed mesh, should be IDENTICAL.
                self.assertEqual(orig, reloaded)


if __name__ == "__main__":
    tests.main()
