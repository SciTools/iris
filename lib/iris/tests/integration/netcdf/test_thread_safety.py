# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests covering thread safety during loading/saving netcdf files.

These tests are intended to catch non-thread-safe behaviour by producing CI
'irregularities' that are noticed and investigated. They cannot reliably
produce standard pytest failures, since the tools for 'correctly'
testing non-thread-safe behaviour are not available at the Python layer.
Thread safety problems can be either produce errors (like a normal test) OR
segfaults (test doesn't complete, pytest-xdiff starts a new group worker, the
end exit code is still non-0), and some problems do not occur in every test
run.

Token assertions are included after the line that is expected to reveal
a thread safety problem, as this seems to be good testing practice.

"""
from pathlib import Path

import dask
from dask import array as da
import numpy as np
import pytest

import iris
from iris.cube import Cube, CubeList
from iris.tests import get_data_path


@pytest.fixture
def tiny_chunks():
    """Guarantee that Dask will use >1 thread by guaranteeing >1 chunk."""

    def _check_tiny_loaded_chunks(cube: Cube):
        assert cube.has_lazy_data()
        cube_lazy_data = cube.core_data()
        assert np.product(cube_lazy_data.chunksize) < cube_lazy_data.size

    with dask.config.set({"array.chunk-size": "1KiB"}):
        yield _check_tiny_loaded_chunks


@pytest.fixture
def save_common(tmp_path):
    save_path = tmp_path / "tmp.nc"

    def _func(cube: Cube):
        assert not save_path.exists()
        iris.save(cube, save_path)
        assert save_path.exists()

    yield _func


@pytest.fixture
def get_cubes_from_netcdf():
    load_dir_path = Path(get_data_path(["NetCDF", "global", "xyt"]))
    loaded = iris.load(load_dir_path.glob("*"), "tcco2")
    smaller = CubeList([c[0] for c in loaded])
    yield smaller


def test_realise_data(tiny_chunks, get_cubes_from_netcdf):
    cube = get_cubes_from_netcdf[0]
    tiny_chunks(cube)
    _ = cube.data  # Any problems are expected here.
    assert not cube.has_lazy_data()


def test_realise_data_multisource(get_cubes_from_netcdf):
    """Load from multiple sources to force Dask to use multiple threads."""
    cubes = get_cubes_from_netcdf
    final_cube = sum(cubes)
    _ = final_cube.data  # Any problems are expected here.
    assert not final_cube.has_lazy_data()


def test_save(tiny_chunks, save_common):
    cube = Cube(da.ones(10000))
    tiny_chunks(cube)
    save_common(cube)  # Any problems are expected here.


def test_stream(tiny_chunks, get_cubes_from_netcdf, save_common):
    cube = get_cubes_from_netcdf[0]
    tiny_chunks(cube)
    save_common(cube)  # Any problems are expected here.


def test_stream_multisource(get_cubes_from_netcdf, save_common):
    """Load from multiple sources to force Dask to use multiple threads."""
    cubes = get_cubes_from_netcdf
    final_cube = sum(cubes)
    save_common(final_cube)  # Any problems are expected here.


def test_stream_multisource__manychunks(
    tiny_chunks, get_cubes_from_netcdf, save_common
):
    """
    As above, but with many more small chunks.

    As this previously showed additional, sporadic problems which only emerge
    (statistically) with larger numbers of chunks.

    """
    cubes = get_cubes_from_netcdf
    final_cube = sum(cubes)
    save_common(final_cube)  # Any problems are expected here.


def test_comparison(get_cubes_from_netcdf):
    """
    Comparing multiple loaded files forces co-realisation.

    See :func:`iris._lazy_data._co_realise_lazy_arrays` .
    """
    cubes = get_cubes_from_netcdf
    _ = cubes[:-1] == cubes[1:]  # Any problems are expected here.
    assert all([c.has_lazy_data() for c in cubes])
