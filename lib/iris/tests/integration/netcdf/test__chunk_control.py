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
from unittest.mock import ANY, Mock

import dask
from dask import distributed
from numpy import dtype
import pytest

import iris
from iris import _lazy_data
from iris.fileformats.netcdf import loader
from iris.fileformats.netcdf.loader import CHUNK_CONTROL
import iris.tests.stock as istk


@pytest.fixture()
def create_cube(tmp_filepath):
    cube = istk.simple_4d_with_hybrid_height()
    cube_varname = "my_var"
    sigma_varname = "my_sigma"
    cube.var_name = cube_varname
    cube.coord("sigma").var_name = sigma_varname
    cube.coord("sigma").guess_bounds()
    iris.save(cube, tmp_filepath)
    yield cube_varname, sigma_varname


@pytest.fixture
def create_file_cube(tmp_filepath):
    iris.save(istk.simple_3d(), tmp_filepath, chunksizes=(1, 3, 4))
    yield None


@pytest.fixture
def tmp_filepath():
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / "tmp.nc"
    yield tmp_path
    shutil.rmtree(tmp_dir)


@pytest.fixture(autouse=True)
def remove_min_bytes():
    old_min_bytes = loader._LAZYVAR_MIN_BYTES
    loader._LAZYVAR_MIN_BYTES = 0
    yield None
    loader._LAZYVAR_MIN_BYTES = old_min_bytes


def test_default(tmp_filepath, create_cube):
    cube = iris.load_cube(tmp_filepath, create_cube[0])
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 4, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (4,)
    assert sigma.lazy_bounds().chunksize == (4, 2)


def test_control_global(tmp_filepath, create_cube):
    with CHUNK_CONTROL.set(model_level_number=2):
        cube = iris.load_cube(tmp_filepath, create_cube[0])
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 2, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (2,)
    assert sigma.lazy_bounds().chunksize == (2, 2)


def test_control_sigma_only(tmp_filepath, create_cube):
    with CHUNK_CONTROL.set(create_cube[1], model_level_number=2):
        cube = iris.load_cube(tmp_filepath, create_cube[0])
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 4, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (2,)
    assert sigma.lazy_bounds().chunksize == (4, 2)


def test_control_cube_var(tmp_filepath, create_cube):
    with CHUNK_CONTROL.set(create_cube[0], model_level_number=2):
        cube = iris.load_cube(tmp_filepath, create_cube[0])
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 2, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (2,)
    assert sigma.lazy_bounds().chunksize == (2, 2)


def test_control_multiple(tmp_filepath, create_cube):
    with CHUNK_CONTROL.set(
        create_cube[0], model_level_number=2
    ), CHUNK_CONTROL.set(create_cube[1], model_level_number=3):
        cube = iris.load_cube(tmp_filepath, create_cube[0])
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 2, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (3,)
    assert sigma.lazy_bounds().chunksize == (2, 2)


def test_neg_one(tmp_filepath, create_cube):
    with CHUNK_CONTROL.set(model_level_number=-1):
        cube = iris.load_cube(tmp_filepath, create_cube[0])
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 4, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (4,)
    assert sigma.lazy_bounds().chunksize == (4, 2)


def test_from_file(tmp_filepath, create_file_cube):
    with CHUNK_CONTROL.from_file():
        cube = iris.load_cube(tmp_filepath)
    assert cube.shape == (2, 3, 4)
    assert cube.lazy_data().chunksize == (1, 3, 4)


def test_as_dask(tmp_filepath, create_cube):
    message = "Mock called, rest of test unneeded"

    loader.as_lazy_data = Mock(side_effect=RuntimeError(message))
    with CHUNK_CONTROL.as_dask():
        try:
            iris.load_cube(tmp_filepath, create_cube[0])
        except RuntimeError as e:
            if str(e) == message:
                pass
            else:
                raise e
    loader.as_lazy_data.assert_called_with(
        ANY, chunks=None, dask_chunking=True
    )


if __name__ == "__main__":
    tests.main()
