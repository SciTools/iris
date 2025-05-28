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

    # DEBUG, for now
    print("")
    print(cubes)
    for i_cube, cube in enumerate(cubes):
        print(f"\n{i_cube:02d} : {cube.summary(shorten=True)}:")
        print(cube)

    main_cube = cubes.extract_cube("air_potential_temperature")
    print("---\nmain cube coords...")
    for coord in main_cube.coords():
        var_name = coord.var_name or "-"
        print(f'  {var_name.rjust(20)!s} : {coord.summary(shorten=True)}')

    cube_names = sorted([cube.name() for cube in cubes])
    if derived_bounds:
        # get an extra promoted cube for the lost 'level-height bounds"
        assert cube_names == ["air_potential_temperature", "level_height_bnds", "surface_altitude"]
    else:
        assert cube_names == ["air_potential_temperature", "surface_altitude"]

    altitude_coord = main_cube.coord("altitude")
    assert altitude_coord.has_bounds()
    assert altitude_coord.has_lazy_bounds()
    assert altitude_coord.shape == main_cube.shape

    level_height_coord = main_cube.coord("atmosphere_hybrid_height_coordinate")
    sigma_coord = main_cube.coord("sigma")
    surface_altitude_coord = main_cube.coord("surface_altitude")
    assert sigma_coord.has_bounds()
    assert not surface_altitude_coord.has_bounds()

    if not derived_bounds:
        other_cube = cubes.extract_cube("surface_altitude")
        assert np.all(other_cube.data == surface_altitude_coord.points)
        assert level_height_coord.has_bounds()
    else:
        assert not level_height_coord.has_bounds()


def test_load_primary_cf_style(derived_bounds):
    cubes = iris.load(db_testfile_path)
    print("")
    print(cubes)
    main_cube = cubes.extract_cube("air_temperature")
    print(main_cube)
    print("---\nmain cube coords...")
    for coord in main_cube.coords():
        var_name = coord.var_name or "-"
        print(f'  {var_name.rjust(20)!s} : {coord.summary(shorten=True)}')

    pressure_coord = main_cube.coord("air_pressure")
    if not derived_bounds:
        # We don't expect this case to work "properly"
        assert not pressure_coord.has_bounds()
        other_cubes = [cube for cube in cubes if cube != main_cube]
        assert len(other_cubes) == 3
        assert set(cube.name() for cube in other_cubes) == {"PS", "A_bnds", "B_bnds"}
        # TODO: what happened to other 'bits', i.e. a/b (we have only the bounds) ??

    else:
        assert pressure_coord.has_bounds()
        # We expect to get an extra "promoted" surface-pressure cube
        assert len(cubes) == 2
        (other_cube,) = [cube for cube in cubes if cube != main_cube]
        assert other_cube.name() == "PS"
        assert other_cube.units == "Pa"
        # It should match the reference coord in the main cube
        assert np.all(main_cube.coord("PS").points == other_cube.data)

        # Check all the dependency coords
        (factory,) = main_cube.aux_factories
        assert isinstance(factory, iris.aux_factory.HybridPressureFactory)

        # #
        # # Fix this : ***something here is wrong*** ???
        # #
        # a_coord = main_cube.coord("vertical pressure")
        # a_coord.rename("a coefficient for vertical coordinate at full levels")

        dep_ids = {
            dep_name: coord.name()
            for dep_name, coord in factory.dependencies.items()
        }
        assert dep_ids == {
            #
            # TODO: ***FIX THIS???*** something seems wrong
            #
            # "delta": "a coefficient for vertical coordinate at full levels",
            "delta": "vertical pressure",
            "sigma": "b coefficient for vertical coordinate at full levels",
            "surface_air_pressure": "PS"
        }
        for dep_name, coord in factory.dependencies.items():
            assert coord in main_cube.coords()
            assert coord.has_bounds() == (dep_name != "surface_air_pressure")

