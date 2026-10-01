# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf.loader.ChunkControl`."""

import re

import dask
import numpy as np
import pytest

import iris
from iris.cube import CubeList
import iris.fileformats.cf
from iris.fileformats.netcdf import loader
import iris.fileformats.netcdf._dataset
from iris.fileformats.netcdf.loader import CHUNK_CONTROL
import iris.tests.stock as istk


@pytest.fixture
def save_cubelist_with_sigma(tmp_filepath):
    cube = istk.simple_4d_with_hybrid_height()
    cube_varname = "my_var"
    sigma_varname = "my_sigma"
    cube.var_name = cube_varname
    cube.coord("sigma").var_name = sigma_varname
    cube.coord("sigma").guess_bounds()
    iris.save(cube, tmp_filepath)
    return cube_varname, sigma_varname


@pytest.fixture
def save_cube_with_chunksize(tmp_filepath):
    cube = istk.simple_3d()
    # adding an aux coord allows us to test that
    # iris.fileformats.netcdf.loader._get_cf_var_data()
    # will only throw an error if from_file mode is
    # True when the entire cube has no specified chunking
    aux = iris.coords.AuxCoord(
        points=np.zeros((3, 4)),
        long_name="random",
        units="1",
    )
    cube.add_aux_coord(aux, [1, 2])
    iris.save(cube, tmp_filepath, chunksizes=(1, 3, 4))


@pytest.fixture(scope="session")
def tmp_filepath(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    tmp_path = tmp_dir / "tmp.nc"
    return str(tmp_path)


@pytest.fixture(autouse=True)
def remove_min_bytes():
    old_min_bytes = loader._LAZYVAR_MIN_BYTES
    loader._LAZYVAR_MIN_BYTES = 0
    yield
    loader._LAZYVAR_MIN_BYTES = old_min_bytes


def test_default(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, _ = save_cubelist_with_sigma
    cubes = CubeList(loader.load_cubes(tmp_filepath))
    cube = cubes.extract_cube(cube_varname)
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 4, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (4,)
    assert sigma.lazy_bounds().chunksize == (4, 2)


def test_netcdf_v3():
    # Just check that it does not fail when loading NetCDF v3 data
    path = iris.tests.get_data_path(
        ["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc.k2"]
    )
    with CHUNK_CONTROL.set(time=-1):
        iris.load(path)


def test_control_global(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, _ = save_cubelist_with_sigma
    with CHUNK_CONTROL.set(model_level_number=2):
        cubes = CubeList(loader.load_cubes(tmp_filepath))
        cube = cubes.extract_cube(cube_varname)
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 2, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (2,)
    assert sigma.lazy_bounds().chunksize == (2, 2)


def test_control_sigma_only(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, sigma_varname = save_cubelist_with_sigma
    with CHUNK_CONTROL.set(sigma_varname, model_level_number=2):
        cubes = CubeList(loader.load_cubes(tmp_filepath))
        cube = cubes.extract_cube(cube_varname)
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 4, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (2,)
    # N.B. this does not apply to bounds array
    assert sigma.lazy_bounds().chunksize == (4, 2)


def test_control_cube_var(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, _ = save_cubelist_with_sigma
    with CHUNK_CONTROL.set(cube_varname, model_level_number=2):
        cubes = CubeList(loader.load_cubes(tmp_filepath))
        cube = cubes.extract_cube(cube_varname)
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 2, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (2,)
    assert sigma.lazy_bounds().chunksize == (2, 2)


def test_invalid_chunksize(tmp_filepath, save_cubelist_with_sigma):
    msg = "'dimension_chunksizes' kwargs should be a dict of `str: int` pairs, not {'model_level_numer': '2'}."
    with pytest.raises(ValueError, match=msg):
        with CHUNK_CONTROL.set(model_level_numer="2"):
            CubeList(loader.load_cubes(tmp_filepath))


def test_invalid_var_name(tmp_filepath, save_cubelist_with_sigma):
    msg = re.escape("'var_names' should be an iterable of strings, not [1, 2].")
    with pytest.raises(ValueError, match=msg):
        with CHUNK_CONTROL.set([1, 2], model_level_numer="2"):
            CubeList(loader.load_cubes(tmp_filepath))


def test_control_multiple(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, sigma_varname = save_cubelist_with_sigma
    with (
        CHUNK_CONTROL.set(cube_varname, model_level_number=2),
        CHUNK_CONTROL.set(sigma_varname, model_level_number=3),
    ):
        cubes = CubeList(loader.load_cubes(tmp_filepath))
        cube = cubes.extract_cube(cube_varname)
    assert cube.shape == (3, 4, 5, 6)
    assert cube.lazy_data().chunksize == (3, 2, 5, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (3,)
    assert sigma.lazy_bounds().chunksize == (2, 2)


def test_neg_one(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, _ = save_cubelist_with_sigma
    with dask.config.set({"array.chunk-size": "50B"}):
        with CHUNK_CONTROL.set(model_level_number=-1):
            cubes = CubeList(loader.load_cubes(tmp_filepath))
            cube = cubes.extract_cube(cube_varname)
    assert cube.shape == (3, 4, 5, 6)
    # uses known good output
    assert cube.lazy_data().chunksize == (1, 4, 1, 1)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (4,)
    assert sigma.lazy_bounds().chunksize == (4, 1)


def test_from_file(tmp_filepath, save_cube_with_chunksize):
    with CHUNK_CONTROL.from_file():
        cube = next(loader.load_cubes(tmp_filepath))
    assert cube.shape == (2, 3, 4)
    assert cube.lazy_data().chunksize == (1, 3, 4)


def test_no_chunks_from_file(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, _ = save_cubelist_with_sigma
    with pytest.raises(KeyError):
        with CHUNK_CONTROL.from_file():
            CubeList(loader.load_cubes(tmp_filepath))


def test_as_dask(tmp_filepath, save_cubelist_with_sigma, mocker):
    """Test as dask.

    No return values, as we can't be sure
    dask chunking behaviour won't change, or that it will differ
    from our own chunking behaviour.
    """
    message = "Mock called, rest of test unneeded"
    as_lazy_data = mocker.patch("iris.fileformats.netcdf._dataset.as_lazy_data")
    as_lazy_data.side_effect = RuntimeError(message)
    with CHUNK_CONTROL.as_dask():
        try:
            CubeList(loader.load_cubes(tmp_filepath))
        except RuntimeError as e:
            if str(e) != message:
                raise e
    as_lazy_data.assert_called_with(
        mocker.ANY,
        meta=mocker.ANY,
        chunks="auto",
        dims_fixed=None,
        cache_key=mocker.ANY,
    )


def test_pinned_optimisation(tmp_filepath, save_cubelist_with_sigma):
    cube_varname, _ = save_cubelist_with_sigma
    with dask.config.set({"array.chunk-size": "250B"}):
        with CHUNK_CONTROL.set(model_level_number=2):
            cubes = CubeList(loader.load_cubes(tmp_filepath))
            cube = cubes.extract_cube(cube_varname)
    assert cube.shape == (3, 4, 5, 6)
    # uses known good output
    # known good output WITHOUT pinning: (1, 1, 5, 6)
    assert cube.lazy_data().chunksize == (1, 2, 2, 6)

    sigma = cube.coord("sigma")
    assert sigma.shape == (4,)
    assert sigma.lazy_points().chunksize == (2,)
    assert sigma.lazy_bounds().chunksize == (2, 2)


class TestChunksFromChunkControl:
    @staticmethod
    def _cf_var(mocker, chunking, shape=(2, 3, 4), cls=None):
        if cls is None:
            cls = iris.fileformats.cf.CFDataVariable
        dimensions = tuple(f"dim_{i}" for i in range(len(shape)))
        cf_data = mocker.MagicMock(
            spec=iris.fileformats.netcdf._dataset.NetCDFDatasetVariable,
            chunking=chunking,
            dimensions=dimensions,
        )
        return mocker.MagicMock(
            spec=cls,
            cf_data=cf_data,
            cf_name="DUMMY_VAR",
            shape=shape,
            dimensions=dimensions,
        )

    def test_as_dask_defers_everything_to_dask(self, mocker):
        cf_var = self._cf_var(mocker, chunking=None)
        with CHUNK_CONTROL.as_dask():
            assert loader._chunks_from_chunk_control(cf_var) == ("auto", None)

    def test_default_unchunked_uses_the_shape(self, mocker):
        cf_var = self._cf_var(mocker, chunking=None)
        chunks, dims_fixed = loader._chunks_from_chunk_control(cf_var)
        assert chunks == [2, 3, 4]
        assert dims_fixed == (None,)

    def test_from_file_adopts_and_fixes_the_store_chunking(self, mocker):
        cf_var = self._cf_var(mocker, chunking=(1, 3, 4))
        with CHUNK_CONTROL.from_file():
            chunks, dims_fixed = loader._chunks_from_chunk_control(cf_var)
        assert chunks == [1, 3, 4]
        assert tuple(bool(flag) for flag in dims_fixed) == (True, True, True)

    def test_from_file_refuses_an_unchunked_data_variable(self, mocker):
        cf_var = self._cf_var(mocker, chunking=None)
        with CHUNK_CONTROL.from_file():
            with pytest.raises(KeyError, match="pre-existing chunk specifications"):
                loader._chunks_from_chunk_control(cf_var)

    def test_from_file_accepts_an_unchunked_coordinate(self, mocker):
        # Only the cube's data variable is required to carry a chunking; an
        # auxiliary coordinate that does not is normal.
        cf_var = self._cf_var(
            mocker,
            chunking=None,
            cls=iris.fileformats.cf.CFAuxiliaryCoordinateVariable,
        )
        with CHUNK_CONTROL.from_file():
            chunks, _ = loader._chunks_from_chunk_control(cf_var)
        assert chunks == [2, 3, 4]


def test_from_file_still_loads_a_small_unchunked_cube(tmp_path, mocker):
    # The KeyError sits behind the "small enough to read whole" shortcut, so a
    # small contiguous cube loads under from_file() today. Hoisting the check
    # into the CF layer, which is where the "is this the data variable" fact
    # lives, would break this - so pin it.
    mocker.patch("iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES", 5000)
    path = str(tmp_path / "small.nc")
    iris.save(istk.simple_3d(), path)
    with CHUNK_CONTROL.from_file():
        cube = next(loader.load_cubes(path))
    assert cube.shape == (2, 3, 4)
    assert not cube.has_lazy_data()
