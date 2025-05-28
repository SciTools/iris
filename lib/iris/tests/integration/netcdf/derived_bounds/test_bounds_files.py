# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""
Integration tests for loading of data with bounds of derived coordinates.

This includes both out "legacy" sample data (which was wrongly recorded), and more
modern testdata (which does not load bounds fully prior to Iris 3.13).

Both "legacy" and "newstyle" loading behaviour needs to be tested, which depends on the
"FUTURE.derived_bounds" flag setting.
"""
import numpy as np
import pytest

import iris
from iris import sample_data_path, load, FUTURE
from iris.aux_factory import HybridHeightFactory

from pathlib import Path
db_testfile_path = Path(__file__).parent / "temp_nc_sources" / "a_new_file.nc"
legacy_filepath = sample_data_path("hybrid_height.nc")


@pytest.fixture(params=[True, False], ids=["with_db", "without_db"])
def derived_bounds(request):
    db = request.param
    with FUTURE.context(derived_bounds=db):
        yield db



def test_load_legacy_hh(derived_bounds):
    cubes = iris.load(legacy_filepath)

    cube_names = sorted([cube.name() for cube in cubes])
    if derived_bounds:
        # get an extra promoted cube for the lost 'level-height bounds"
        expected_cube_names = ["air_potential_temperature", "level_height_bnds", "surface_altitude"]
    else:
        expected_cube_names = ["air_potential_temperature", "surface_altitude"]
    assert cube_names == expected_cube_names

    main_cube = cubes.extract_cube("air_potential_temperature")
    altitude_coord = main_cube.coord("altitude")
    assert altitude_coord.has_bounds()
    assert altitude_coord.has_lazy_bounds()
    assert altitude_coord.shape == main_cube.shape

    level_height_coord = main_cube.coord("atmosphere_hybrid_height_coordinate")
    sigma_coord = main_cube.coord("sigma")
    surface_altitude_coord = main_cube.coord("surface_altitude")
    assert sigma_coord.has_bounds()
    assert not surface_altitude_coord.has_bounds()
    surface_altitude_cube = cubes.extract_cube("surface_altitude")
    assert np.all(surface_altitude_cube.data == surface_altitude_coord.points)

    if not derived_bounds:
        assert level_height_coord.has_bounds()
    else:
        assert not level_height_coord.has_bounds()


def test_load_primary_cf_style(derived_bounds):
    cubes = iris.load(db_testfile_path)

    cube_names = sorted([cube.name() for cube in cubes])
    if derived_bounds:
        expected_cube_names = ["PS", "air_temperature"]
    else:
        # In this case, the bounds are not properly connected
        expected_cube_names = ["A_bnds", "B_bnds", "PS", "air_temperature"]
    assert cube_names == expected_cube_names

    # Check all the coords on the cube, including whether they have bounds
    main_cube = cubes.extract_cube("air_temperature")
    assert set(co.name() for co in main_cube.coords()) == set([
        "a coefficient for vertical coordinate at full levels",
        "b coefficient for vertical coordinate at full levels",
        "atmosphere_hybrid_sigma_pressure_coordinate",
        "vertical pressure",
        "PS",
        "air_pressure",
        "P0",
    ])

    # First, the main hybrid coord
    pressure_coord = main_cube.coord("air_pressure")
    assert pressure_coord.has_bounds() == derived_bounds
    assert pressure_coord.var_name is None
    assert main_cube.coord_dims(pressure_coord) == (0, 1, 2)

    co_a = main_cube.coord("a coefficient for vertical coordinate at full levels")
    assert co_a.var_name == "A"
    assert co_a.has_bounds() == derived_bounds
    assert main_cube.coord_dims(co_a) == (0,)

    co_b = main_cube.coord("b coefficient for vertical coordinate at full levels")
    assert co_b.var_name == "B"
    assert co_b.has_bounds() == derived_bounds
    assert main_cube.coord_dims(co_b) == (0,)

    co_eta = main_cube.coord("atmosphere_hybrid_sigma_pressure_coordinate")
    assert co_eta.var_name == "eta"
    assert co_eta.has_bounds()
    assert main_cube.coord_dims(co_eta) == (0,)

    # N.B. this coord is 'made up' by the factory, and does *not* come from a variable
    co_VP = main_cube.coord("vertical pressure")
    assert co_VP.var_name == "ap"
    assert co_VP.has_bounds() == derived_bounds
    assert main_cube.coord_dims(co_VP) == (0,)

    # This is the surface pressure
    co_PS = main_cube.coord("PS")
    assert co_PS.var_name == "PS"
    assert not co_PS.has_bounds()
    assert main_cube.coord_dims(co_PS) == (1, 2)

    # The scalar reference
    co_P0 = main_cube.coord("P0")
    assert co_P0.var_name == "P0"
    assert not co_P0.has_bounds()
    assert main_cube.coord_dims(co_P0) == ()


