# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.tests.stock.netcdf` module."""

from iris import load_cube
from iris.mesh import MeshCoord, MeshXY
from iris.tests.stock import netcdf


class XIOSFileMixin:
    def create_synthetic_file(self, **create_kwargs):
        # Should be overridden to invoke one of the create_file_ functions.
        # E.g.
        # return netcdf.create_file__xios_2d_face_half_levels(
        #     temp_file_dir=self.temp_dir, dataset_name="mesh", **create_kwargs
        # )
        raise NotImplementedError

    def create_synthetic_test_cube(self, **create_kwargs):
        file_path = self.create_synthetic_file(**create_kwargs)
        cube = load_cube(file_path)
        return cube

    def check_cube(self, cube, shape, location, level):
        # Basic checks on the primary data cube.
        assert cube.var_name == "thing"
        assert cube.long_name == "thingness"
        assert cube.shape == shape

        # Also a few checks on the attached mesh-related information.
        last_dim = cube.ndim - 1
        assert isinstance(cube.mesh, MeshXY)
        assert cube.mesh_dim() == last_dim
        assert cube.location == location
        for coord_name in ("longitude", "latitude"):
            coord = cube.coord(coord_name)
            assert isinstance(coord, MeshCoord)
            assert coord.shape == (shape[last_dim],)
        assert cube.mesh.var_name.endswith(f"{level}_levels")


class Test_create_file__xios_2d_face_half_levels(XIOSFileMixin):
    def create_synthetic_file(self, **create_kwargs):
        return netcdf.create_file__xios_2d_face_half_levels(
            dataset_name="mesh", **create_kwargs
        )

    def test_basic_load(self, tmp_path):
        cube = self.create_synthetic_test_cube(temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 866), location="face", level="half")

    def test_scale_mesh(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_faces=10, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 10), location="face", level="half")

    def test_scale_time(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_times=3, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(3, 866), location="face", level="half")


class Test_create_file__xios_3d_face_half_levels(XIOSFileMixin):
    def create_synthetic_file(self, **create_kwargs):
        return netcdf.create_file__xios_3d_face_half_levels(
            dataset_name="mesh", **create_kwargs
        )

    def test_basic_load(self, tmp_path):
        cube = self.create_synthetic_test_cube(temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 38, 866), location="face", level="half")

    def test_scale_mesh(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_faces=10, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 38, 10), location="face", level="half")

    def test_scale_time(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_times=3, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(3, 38, 866), location="face", level="half")

    def test_scale_levels(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_levels=10, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 10, 866), location="face", level="half")


class Test_create_file__xios_3d_face_full_levels(XIOSFileMixin):
    def create_synthetic_file(self, **create_kwargs):
        return netcdf.create_file__xios_3d_face_full_levels(
            dataset_name="mesh", **create_kwargs
        )

    def test_basic_load(self, tmp_path):
        cube = self.create_synthetic_test_cube(temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 39, 866), location="face", level="full")

    def test_scale_mesh(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_faces=10, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 39, 10), location="face", level="full")

    def test_scale_time(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_times=3, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(3, 39, 866), location="face", level="full")

    def test_scale_levels(self, tmp_path):
        cube = self.create_synthetic_test_cube(n_levels=10, temp_file_dir=tmp_path)
        self.check_cube(cube, shape=(1, 10, 866), location="face", level="full")
