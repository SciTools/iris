# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
import shutil
import tempfile

import iris
from iris.fileformats.netcdf.loader import CHUNK_CONTROL
from iris.fileformats.netcdf import loader
import iris.tests.stock as istk


class TestChunking(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        cls.old_min_bytes = loader._LAZYVAR_MIN_BYTES
        loader._LAZYVAR_MIN_BYTES = 0
        cls.temp_dir = tempfile.mkdtemp()
        cube = istk.simple_4d_with_hybrid_height()
        cls.cube = cube
        cls.cube_varname = "my_var"
        cls.sigma_varname = "my_sigma"
        cube.var_name = cls.cube_varname
        cube.coord("sigma").var_name = cls.sigma_varname
        cube.coord("sigma").guess_bounds()
        cls.tempfile_path = Path(cls.temp_dir) / "tmp.nc"
        iris.save(cls.cube, cls.tempfile_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        loader._LAZYVAR_MIN_BYTES = cls.old_min_bytes

    def test_default(self):
        cube = iris.load_cube(self.tempfile_path, self.cube_varname)
        self.assertEqual((3, 4, 5, 6), cube.shape)
        self.assertEqual((3, 4, 5, 6), cube.lazy_data().chunksize)
        sigma = cube.coord("sigma")
        self.assertEqual((4,), sigma.shape)
        self.assertEqual((4,), sigma.lazy_points().chunksize)
        self.assertEqual((4, 2), sigma.lazy_bounds().chunksize)

    def test_control_global(self):
        with CHUNK_CONTROL.set(model_level_number=2):
            cube = iris.load_cube(self.tempfile_path, self.cube_varname)
        self.assertEqual((3, 4, 5, 6), cube.shape)
        self.assertEqual((3, 2, 5, 6), cube.lazy_data().chunksize)
        sigma = cube.coord("sigma")
        self.assertEqual((4,), sigma.shape)
        self.assertEqual((2,), sigma.lazy_points().chunksize)
        self.assertEqual((2, 2), sigma.lazy_bounds().chunksize)

    def test_control_sigma_only(self):
        with CHUNK_CONTROL.set(self.sigma_varname, model_level_number=2):
            cube = iris.load_cube(self.tempfile_path, self.cube_varname)
        self.assertEqual((3, 4, 5, 6), cube.shape)
        self.assertEqual((3, 4, 5, 6), cube.lazy_data().chunksize)
        sigma = cube.coord("sigma")
        self.assertEqual((4,), sigma.shape)
        self.assertEqual((2,), sigma.lazy_points().chunksize)
        # N.B. this does not apply to bounds array
        self.assertEqual((4, 2), sigma.lazy_bounds().chunksize)

    def test_control_cube_var(self):
        with CHUNK_CONTROL.set(self.cube_varname, model_level_number=2):
            cube = iris.load_cube(self.tempfile_path, self.cube_varname)
        self.assertEqual((3, 4, 5, 6), cube.shape)
        self.assertEqual((3, 2, 5, 6), cube.lazy_data().chunksize)
        sigma = cube.coord("sigma")
        self.assertEqual((4,), sigma.shape)
        self.assertEqual((2,), sigma.lazy_points().chunksize)
        # N.B. this does not apply to bounds array
        self.assertEqual((2, 2), sigma.lazy_bounds().chunksize)

    def test_control_multiple(self):
        with CHUNK_CONTROL.set(
            self.cube_varname, model_level_number=2
        ), CHUNK_CONTROL.set(self.sigma_varname, model_level_number=3):
            cube = iris.load_cube(self.tempfile_path, self.cube_varname)
        self.assertEqual((3, 4, 5, 6), cube.shape)
        self.assertEqual((3, 2, 5, 6), cube.lazy_data().chunksize)
        sigma = cube.coord("sigma")
        self.assertEqual((4,), sigma.shape)
        self.assertEqual((3,), sigma.lazy_points().chunksize)
        self.assertEqual((2, 2), sigma.lazy_bounds().chunksize)


if __name__ == "__main__":
    tests.main()
