# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test loading PP data with time-varying orography."""

import pytest

import iris
from iris import LOAD_POLICY, sample_data_path


@pytest.fixture(params=["default", "recommended", "legacy"])
def load_policy(request):
    return request.param


def test_load_pp_timevarying_orography(load_policy):
    testdata_dirpath = sample_data_path("time_varying_hybrid_height", "*.pp")

    with LOAD_POLICY.context(load_policy):
        cubes = iris.load(testdata_dirpath)

    n_cubes = len(cubes)
    if load_policy == "legacy":
        # This doesn't merge fully: get a phenomenon cube for each reference field
        assert n_cubes == 4
    else:
        # Other load policies load with full merge, producing a 4D result.
        assert n_cubes == 2
        phenom_cube = cubes.extract_cube("x_wind")
        ref_cube = cubes.extract_cube("surface_altitude")

        cube_dims = [
            phenom_cube.coord(dim_coords=True, dimensions=i_dim).name()
            for i_dim in range(phenom_cube.ndim)
        ]
        assert cube_dims == ["model_level_number", "time", "latitude", "longitude"]

        ref_coord = phenom_cube.coord("surface_altitude")
        ref_coord_dims = [
            phenom_cube.coord(dim_coords=True, dimensions=i_dim).name()
            for i_dim in phenom_cube.coord_dims(ref_coord)
        ]
        assert ref_coord_dims == ["time", "latitude", "longitude"]

        ref_cube_dims = [
            ref_cube.coord(dim_coords=True, dimensions=i_dim).name()
            for i_dim in range(ref_cube.ndim)
        ]
        assert ref_cube_dims == ref_cube_dims

        derived_coord = phenom_cube.coord("altitude")
        derived_dims = [
            phenom_cube.coord(dim_coords=True, dimensions=i_dim).name()
            for i_dim in phenom_cube.coord_dims(derived_coord)
        ]
        assert derived_dims == ["model_level_number", "time", "latitude", "longitude"]
