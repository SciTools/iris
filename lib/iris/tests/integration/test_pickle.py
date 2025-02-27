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
from iris.tests import MIN_PICKLE_PROTOCOL

TESTED_PROTOCOLS = list(range(MIN_PICKLE_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1))


def pickle_cube(path, protocol, filename):
    # Ensure that data proxies are pickleable.
    cube = iris.load(path)[0]
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
