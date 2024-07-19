# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.tests.stock.netcdf` module."""

import shutil
import tempfile

from iris import load_cube
from iris.experimental.ugrid.load import PARSE_UGRID_ON_LOAD
from iris.experimental.ugrid.mesh import MeshCoord, MeshXY

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
from iris.tests.stock import netcdf


class XIOSFileMixin(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        # Create a temp directory for transient test files.
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        # Destroy the temp directory.
        shutil.rmtree(cls.temp_dir)

    def create_synthetic_file(self, **create_kwargs):
        # Should be overridden to invoke one of the create_file_ functions.
        # E.g.
        # return netcdf.create_file__xios_2d_face_half_levels(
        #     temp_file_dir=self.temp_dir, dataset_name="mesh", **create_kwargs
        # )
        raise NotImplementedError

    def create_synthetic_test_cube(self, **create_kwargs):
        file_path = self.create_synthetic_file(**create_kwargs)
        with PARSE_UGRID_ON_LOAD.context():
            cube = load_cube(file_path)
        return cube

    def check_cube(self, cube, shape, location, level):
        # Basic checks on the primary data cube.
        self.assertEqual(cube.var_name, "thing")
        self.assertEqual(cube.long_name, "thingness")
        self.assertEqual(cube.shape, shape)

        # Also a few checks on the attached mesh-related information.
        last_dim = cube.ndim - 1
        self.assertIsInstance(cube.mesh, MeshXY)
        self.assertEqual(cube.mesh_dim(), last_dim)
        self.assertEqual(cube.location, location)
        for coord_name in ("longitude", "latitude"):
            coord = cube.coord(coord_name)
            self.assertIsInstance(coord, MeshCoord)
            self.assertEqual(coord.shape, (shape[last_dim],))
        self.assertTrue(cube.mesh.var_name.endswith(f"{level}_levels"))


class Test_create_file__xios_2d_face_half_levels(XIOSFileMixin):
    def create_synthetic_file(self, **create_kwargs):
        return netcdf.create_file__xios_2d_face_half_levels(
            temp_file_dir=self.temp_dir, dataset_name="mesh", **create_kwargs
        )

    def test_basic_load(self):
        cube = self.create_synthetic_test_cube()
        self.check_cube(cube, shape=(1, 866), location="face", level="half")

    def test_scale_mesh(self):
        cube = self.create_synthetic_test_cube(n_faces=10)
        self.check_cube(cube, shape=(1, 10), location="face", level="half")

    def test_scale_time(self):
        cube = self.create_synthetic_test_cube(n_times=3)
        self.check_cube(cube, shape=(3, 866), location="face", level="half")


class Test_create_file__xios_3d_face_half_levels(XIOSFileMixin):
    def create_synthetic_file(self, **create_kwargs):
        return netcdf.create_file__xios_3d_face_half_levels(
            temp_file_dir=self.temp_dir, dataset_name="mesh", **create_kwargs
        )

    def test_basic_load(self):
        cube = self.create_synthetic_test_cube()
        self.check_cube(cube, shape=(1, 38, 866), location="face", level="half")

    def test_scale_mesh(self):
        cube = self.create_synthetic_test_cube(n_faces=10)
        self.check_cube(cube, shape=(1, 38, 10), location="face", level="half")

    def test_scale_time(self):
        cube = self.create_synthetic_test_cube(n_times=3)
        self.check_cube(cube, shape=(3, 38, 866), location="face", level="half")

    def test_scale_levels(self):
        cube = self.create_synthetic_test_cube(n_levels=10)
        self.check_cube(cube, shape=(1, 10, 866), location="face", level="half")


class Test_create_file__xios_3d_face_full_levels(XIOSFileMixin):
    def create_synthetic_file(self, **create_kwargs):
        return netcdf.create_file__xios_3d_face_full_levels(
            temp_file_dir=self.temp_dir, dataset_name="mesh", **create_kwargs
        )

    def test_basic_load(self):
        cube = self.create_synthetic_test_cube()
        self.check_cube(cube, shape=(1, 39, 866), location="face", level="full")

    def test_scale_mesh(self):
        cube = self.create_synthetic_test_cube(n_faces=10)
        self.check_cube(cube, shape=(1, 39, 10), location="face", level="full")

    def test_scale_time(self):
        cube = self.create_synthetic_test_cube(n_times=3)
        self.check_cube(cube, shape=(3, 39, 866), location="face", level="full")

    def test_scale_levels(self):
        cube = self.create_synthetic_test_cube(n_levels=10)
        self.check_cube(cube, shape=(1, 10, 866), location="face", level="full")


if __name__ == "__main__":
    tests.main()
