# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for pickling things."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
import pickle

import pytest

import iris
from iris.cube import Cube
from iris.tests import MIN_PICKLE_PROTOCOL
from iris.tests.stock import simple_2d
from iris.tests.stock.mesh import sample_mesh_cube

TESTED_PROTOCOLS = list(range(MIN_PICKLE_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1))


def pickle_cube(cube_or_path, protocol, filename):
    # Ensure that data proxies are pickleable.
    if isinstance(cube_or_path, Cube):
        cube = cube_or_path
    else:
        cube = iris.load(cube_or_path)[0]
    with open(filename, "wb") as f:
        pickle.dump(cube, f, protocol)
    with open(filename, "rb") as f:
        ncube = pickle.load(f)
    assert ncube == cube


@pytest.mark.parametrize("protocol", TESTED_PROTOCOLS)
@tests.skip_data
def test_netcdf(protocol, tmp_path):
    path = tests.get_data_path(
        ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
    )
    tmp_file = tmp_path / "netcdf.pkl"
    pickle_cube(path, protocol, tmp_file)


@pytest.mark.parametrize("protocol", TESTED_PROTOCOLS)
@tests.skip_data
def test_pp(protocol, tmp_path):
    path = tests.get_data_path(("PP", "aPPglob1", "global.pp"))
    tmp_file = tmp_path / "pp.pkl"
    pickle_cube(path, protocol, tmp_file)


@pytest.mark.parametrize("protocol", TESTED_PROTOCOLS)
@tests.skip_data
def test_ff(protocol, tmp_path):
    path = tests.get_data_path(("FF", "n48_multi_field"))
    tmp_file = tmp_path / "ff.pkl"
    pickle_cube(path, protocol, tmp_file)


@pytest.mark.parametrize("protocol", TESTED_PROTOCOLS)
@pytest.mark.parametrize("cubetype", ["regular", "mesh"])
def test_synthetic(protocol, cubetype, tmp_path):
    """Check that simple cubes can be pickled, including mesh cubes."""
    source_fn = {"regular": simple_2d, "mesh": sample_mesh_cube}[cubetype]
    test_cube = source_fn()
    tmp_filepath = tmp_path / "tmp.pkl"
    pickle_cube(test_cube, protocol, tmp_filepath)
