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
        print(f'  {coord.var_name.rjust(20)!s} : {coord.summary(shorten=True)}')

    assert len(cubes) == 2
    (other_cube,) = [cube for cube in cubes if cube != main_cube]

    if not derived_bounds:
        assert other_cube.name() == "surface_altitude"
        altitude_coord = main_cube.coord("altitude")
        assert np.all(other_cube.data == main_cube.coord("surface_altitude").points)
    else:
        assert not main_cube.coords("altitude")

        # #
        # # OK, confirm for now
        # #   but ***THIS BIT*** is surely wrong ????
        # #
        # assert other_cube.name() == "level_height_bnds"
        #
        # # FOR NOW: fix our "problem" by adding a factory "manually"
        # sigma = main_cube.coord("sigma")
        # delta = main_cube.coord("atmosphere_hybrid_height_coordinate")
        # orog = main_cube.coord("surface_altitude")
        # fact = HybridHeightFactory(sigma=sigma, delta=delta, orography=orog)
        # main_cube.add_aux_factory(fact)
        # assert main_cube.coords("altitude")
        # altitude_coord = main_cube.coord("altitude")

    assert altitude_coord.has_bounds()
    assert altitude_coord.has_lazy_bounds()
    assert altitude_coord.shape == main_cube.shape

    # Also confirm what we expect from the other factory components (dependencies)
    (factory,) = main_cube.aux_factories
    for coord in factory.dependencies.values():
        assert coord in main_cube.coords()
        if coord.name() in ("atmosphere_hybrid_height_coordinate", "sigma"):
            assert coord.has_bounds()
        else:
            assert coord.name() == "surface_altitude"
            assert not coord.has_bounds()


def test_load_primary_cf_style(derived_bounds):
    cubes = iris.load(db_testfile_path)
    print("")
    print(cubes)
    main_cube = cubes.extract_cube("air_temperature")
    print(main_cube)
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

