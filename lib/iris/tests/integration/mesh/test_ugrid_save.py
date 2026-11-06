# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for NetCDF-UGRID file saving."""

import glob
from pathlib import Path

import pytest

import iris
import iris.fileformats.netcdf
from iris.tests import _shared_utils
from iris.tests.stock.netcdf import _add_standard_data, ncgen_from_cdl


class TestBasicSave:
    @pytest.fixture(autouse=True, scope="class")
    def _setup_class(self, request, tmp_path_factory):
        cls = request.cls
        cls.temp_dir = tmp_path_factory.mktemp("test_basic_save")
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

    def test_example_result_cdls(self, request):
        # Snapshot the result of saving the example cases.
        for ex_name, cdl_path in self.example_names_paths.items():
            # Create a test netcdf file.
            target_ncfile_path = str(self.temp_dir / f"{ex_name}.nc")
            ncgen_from_cdl(cdl_str=None, cdl_path=cdl_path, nc_path=target_ncfile_path)
            # Fill in blank data-variables.
            _add_standard_data(target_ncfile_path)
            # Load as Iris data
            cubes = iris.load(target_ncfile_path)
            # Re-save, to check the save behaviour.
            resave_ncfile_path = str(self.temp_dir / f"{ex_name}_resaved.nc")
            iris.save(cubes, resave_ncfile_path)
            # Check the output against a CDL snapshot.
            refdir_relpath = "integration/experimental/ugrid_save/TestBasicSave/"
            reffile_name = str(Path(cdl_path).name).replace(".nc", ".cdl")
            reffile_path = refdir_relpath + reffile_name
            _shared_utils.assert_CDL(
                request, resave_ncfile_path, reference_filename=reffile_path
            )

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
            orig_cubes = iris.load(target_ncfile_path)

            if "ex4" in ex_name:
                # Discard the extra formula terms component cubes
                # Saving these does not do what you expect
                orig_cubes = orig_cubes.extract("datavar")

            # Save-and-load-back to compare the Iris saved result.
            resave_ncfile_path = str(self.temp_dir / f"{ex_name}_resaved.nc")
            iris.save(orig_cubes, resave_ncfile_path)
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
                assert orig.location == reloaded.location
                orig_mesh = orig.mesh
                reloaded_mesh = reloaded.mesh
                assert orig_mesh.all_coords == reloaded_mesh.all_coords
                assert orig_mesh.all_connectivities == reloaded_mesh.all_connectivities
                # Index the cubes to replace meshes with meshcoord-derived aux coords.
                # This needs [:0] on the mesh dim, so do that on all dims.
                keys = tuple([slice(0, None)] * orig.ndim)
                orig = orig[keys]
                reloaded = reloaded[keys]
                # Resulting cubes, with collapsed mesh, should be IDENTICAL.
                assert orig == reloaded
