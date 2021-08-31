# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris.experimental.ugrid.load_mesh` function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD, load_mesh


class Tests(tests.IrisTest):
    # All 'real' tests have been done for load_meshes(). Here we just check
    #  that load_mesh() works with load_meshes() correctly, using mocking.
    def setUp(self):
        self.load_meshes_mock = self.patch(
            "iris.experimental.ugrid.load_meshes"
        )
        # The expected return from load_meshes - a dict of files, each with
        #  a list of meshes.
        self.load_meshes_mock.return_value = {"file": ["mesh"]}

    def test_calls_load_meshes(self):
        args = [("file_1", "file_2"), "my_var_name"]
        with PARSE_UGRID_ON_LOAD.context():
            _ = load_mesh(args)
        self.assertTrue(self.load_meshes_mock.called_with(args))

    def test_returns_mesh(self):
        with PARSE_UGRID_ON_LOAD.context():
            mesh = load_mesh([])
        self.assertEqual(mesh, "mesh")

    def test_single_mesh(self):
        # Override the load_meshes_mock return values to provoke errors.
        def common(ret_val):
            self.load_meshes_mock.return_value = ret_val
            with self.assertRaisesRegex(ValueError, "Expecting 1 mesh.*"):
                with PARSE_UGRID_ON_LOAD.context():
                    _ = load_mesh([])

        # Too many.
        common({"file": ["mesh1", "mesh2"]})
        # Too few.
        common({"file": []})
