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

MIN_PICKLE_PROTOCOL = 4
TESTED_PROTOCOLS = list(range(MIN_PICKLE_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1))


def pickle_cube(path, protocol):
    # Ensure that data proxies are pickleable.
    cube = iris.load(path)[0]
    with self.temp_filename(".pkl") as filename:
        with open(filename, "wb") as f:
            pickle.dump(cube, f, protocol)
        with open(filename, "rb") as f:
            ncube = pickle.load(f)
    assert ncube == cube


@pytest.mark.parametrize("protocol", TESTED_PROTOCOLS)
@tests.skip_data
def test_netcdf(protocol):
    path = tests.get_data_path(
        ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
    )
    pickle_cube(path, protocol)


@pytest.mark.parametrize("protocol", TESTED_PROTOCOLS)
@tests.skip_data
def test_pp(protocol):
    path = tests.get_data_path(("PP", "aPPglob1", "global.pp"))
    pickle_cube(path, protocol)


@pytest.mark.parametrize("protocol", TESTED_PROTOCOLS)
@tests.skip_data
def test_ff(protocol):
    self.path = tests.get_data_path(("FF", "n48_multi_field"))
    pickle_cube(path, protocol)
