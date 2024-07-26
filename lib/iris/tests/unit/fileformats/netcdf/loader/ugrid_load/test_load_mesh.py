# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.mesh.load_mesh` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.fileformats.netcdf.ugrid_load import load_mesh


class Tests(tests.IrisTest):
    # All 'real' tests have been done for load_meshes(). Here we just check
    #  that load_mesh() works with load_meshes() correctly, using mocking.
    def setUp(self):
        tgt = "iris.fileformats.netcdf.ugrid_load.load_meshes"
        self.load_meshes_mock = self.patch(tgt)
        # The expected return from load_meshes - a dict of files, each with
        #  a list of meshes.
        self.load_meshes_mock.return_value = {"file": ["mesh"]}

    def test_calls_load_meshes(self):
        args = [("file_1", "file_2"), "my_var_name"]
        _ = load_mesh(args)
        assert self.load_meshes_mock.call_count == 1
        assert self.load_meshes_mock.call_args == ((args, None),)

    def test_returns_mesh(self):
        mesh = load_mesh([])
        self.assertEqual(mesh, "mesh")

    def test_single_mesh(self):
        # Override the load_meshes_mock return values to provoke errors.
        def common(ret_val):
            self.load_meshes_mock.return_value = ret_val
            with self.assertRaisesRegex(ValueError, "Expecting 1 mesh.*"):
                _ = load_mesh([])

        # Too many.
        common({"file": ["mesh1", "mesh2"]})
        # Too few.
        common({"file": []})
