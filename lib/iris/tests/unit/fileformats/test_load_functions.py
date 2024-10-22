# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for iris load functions.

* :func:`iris.load`
* :func:`iris.load_cube`
* :func:`iris.load_cubes`
* :func:`iris.load_raw`
"""

import re
from typing import Iterable
from unittest import mock

import numpy as np
import pytest

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

_time_unit = "days since 2001-01-01"


def cu(n="a", t=0, z=0):
    """Create a single test cube.

    All cubes have, potentially, 4 dimensions (z, t, y, x).
    The (y, x) dims are always the same, but (z, t) can  be scalar, or various lengths.
    t/z values which are scalar/vector produce likewise scalar/vector coordinates.
    """
    yco = DimCoord(np.arange(3), long_name="latitude", units="degrees")
    xco = DimCoord(np.arange(4), long_name="longitude", units="degrees")
    dim_coords = [yco, xco]
    shape = [3, 4]  # the xy shape
    scalar_coords = []
    tco = DimCoord(
        np.array(t, dtype=np.float32), standard_name="time", units=_time_unit
    )
    zco = DimCoord(np.array(z, dtype=np.float32), standard_name="height", units="m")
    for tz, tzco in [(t, tco), (z, zco)]:
        if isinstance(tz, Iterable):
            # N.B. insert an extra dim at the front
            dim_coords[:0] = [tzco]
            shape[:0] = tzco.shape[:1]
        else:
            scalar_coords.append(tzco)

    cube = Cube(
        data=np.zeros(shape),
        long_name=n,
        dim_coords_and_dims=[(dim, i_dim) for i_dim, dim in enumerate(dim_coords)],
        aux_coords_and_dims=[(dim, ()) for dim in scalar_coords],
    )
    return cube


@pytest.fixture(params=["load", "load_cube", "load_cubes", "load_raw"])
def loadfunc_name(request):
    # N.B. "request" is a standard PyTest fixture
    return request.param  # Return the name of the attribute to test.


def run_testcase(input_cubes, loadfunc_name, constraints=None):
    loadfunc = getattr(iris, loadfunc_name)

    def mock_generate_cubes(uris, callback, constraints):
        for cube in input_cubes:
            yield cube

    try:
        with mock.patch("iris._generate_cubes", mock_generate_cubes):
            result = loadfunc(input_cubes, constraints)
    except Exception as e:
        result = e

    return result


def debug_result(cubes):
    print()
    print(cubes)
    if isinstance(cubes, iris.cube.CubeList):
        print(len(cubes), " cubes..")
        for i_cube, cube in enumerate(cubes):
            vh = cube.coord("height").points
            vt = cube.coord("time").points
            print(i_cube, cube.name(), ": h=", vh, " :: t=", vt)


def check_result(input_cubes, loadfunc_name, result, expected_results):
    if "load_raw" not in expected_results and loadfunc_name == "load_raw":
        expected = input_cubes
    else:
        expected = expected_results[loadfunc_name]

    if isinstance(expected, str):
        # We expect an error result : stored 'expected' is a regexp to match its repr
        assert re.search(expected, repr(result))
    else:
        assert result == expected


class TestLoadFunctions:
    def test_mergeable(self, loadfunc_name):
        _cube = cu(t=(0, 1), z=(0, 1))
        input_cubes = [cu(t=i_t, z=i_z) for i_t in (0, 1) for i_z in (0, 1)]
        expected_results = {
            "load": [_cube],
            "load_cube": _cube,
            "load_cubes": [_cube],
        }
        result = run_testcase(input_cubes, loadfunc_name)
        check_result(input_cubes, loadfunc_name, result, expected_results)

    def test_multiple(self, loadfunc_name):
        input_cubes = [cu(), cu(n="b")]
        expected_results = {
            "load": [cu(), cu(n="b")],
            "load_cube": "ConstraintMismatchError.*failed to merge into a single cube",
            "load_cubes": r"ConstraintMismatchError.*-> \d+ cubes",
        }
        result = run_testcase(input_cubes, loadfunc_name)
        check_result(input_cubes, loadfunc_name, result, expected_results)

    def test_multiple_constrained(self, loadfunc_name):
        cube, cube_b = cu(), cu(n="b")
        input_cubes = [cube, cube_b]
        constraint = "a"
        expected_results = {
            "load": [cube],
            "load_cube": cube,
            "load_cubes": [cube],
            "load_raw": [cube],
        }
        result = run_testcase(input_cubes, loadfunc_name, constraints=constraint)
        check_result(input_cubes, loadfunc_name, result, expected_results)

    def test_multiple_multi_constraints(self, loadfunc_name):
        ca, cb, cc = cu(), cu(n="b"), cu(n="c")
        input_cubes = [ca, cb, cc]
        constraints = ["c", "a"]
        expected_results = {
            "load": [cc, ca],
            "load_cube": "ValueError.*only a single constraint is allowed",
            "load_cubes": [cc, ca],
            "load_raw": [cc, ca],
        }
        result = run_testcase(input_cubes, loadfunc_name, constraints=constraints)
        check_result(input_cubes, loadfunc_name, result, expected_results)

    def test_nonmergeable_part_missing(self, loadfunc_name):
        c1, c2, c3, c4 = [cu(t=i_t, z=i_z) for i_t in (0, 1) for i_z in (0, 1)]
        input_cubes = [c1, c2, c4]

        c124 = cu(t=(0, 1, 2))
        c124.remove_coord("time")  # we now have an unnamed dimension
        c124.remove_coord("height")  # we now have an unnamed dimension
        c124.add_aux_coord(AuxCoord([0.0, 1, 1], standard_name="height", units="m"), 0)
        c124.add_aux_coord(
            AuxCoord([0.0, 0, 1], standard_name="time", units=_time_unit), 0
        )
        expected_results = {
            "load": [c124],
            "load_cube": c124,
            "load_cubes": [c124],
        }
        result = run_testcase(input_cubes, loadfunc_name)
        check_result(input_cubes, loadfunc_name, result, expected_results)

    def test_nonmergeable_part_extra(self, loadfunc_name):
        c1, c2, c3, c4 = [cu(t=i_t, z=i_z) for i_t in (0, 1) for i_z in (0, 1)]
        c5 = cu(t=5)
        input_cubes = [c1, c2, c5, c4, c3]  # scramble order, just to test

        cx = cu(t=range(5))
        cx.remove_coord("time")  # we now have an unnamed dimension
        cx.remove_coord("height")  # we now have an unnamed dimension
        cx.add_aux_coord(
            AuxCoord([0.0, 1, 0, 1, 0], standard_name="height", units="m"), 0
        )
        cx.add_aux_coord(
            AuxCoord([0.0, 0, 5, 1, 1], standard_name="time", units=_time_unit), 0
        )
        expected_results = {
            "load": [cx],
            "load_cube": cx,
            "load_cubes": [cx],
        }
        result = run_testcase(input_cubes, loadfunc_name)
        check_result(input_cubes, loadfunc_name, result, expected_results)

    def test_constraint_overlap(self, loadfunc_name):
        c1, c2, c3, c4, c5, c6 = (cu(z=ind) for ind in (1, 2, 3, 4, 5, 6))
        input_cubes = [c1, c2, c3, c4, c5, c6]
        constraints = [
            iris.Constraint(height=[1, 2]),
            iris.Constraint(height=[1, 4, 5]),
        ]
        c12 = cu(z=[1, 2])
        c145 = cu(z=[1, 4, 5])
        expected_results = {
            "load": [c12, c145],
            "load_cube": "ValueError.*only a single constraint is allowed",
            "load_cubes": [c12, c145],  # selected parts merge, as for load
            "load_raw": [c1, c2, c1, c4, c5],  # THIS VERY STRANGE BEHAVIOUR!!
        }
        result = run_testcase(input_cubes, loadfunc_name, constraints=constraints)
        check_result(input_cubes, loadfunc_name, result, expected_results)

    def test_multiple_match(self, loadfunc_name):
        c1 = cu(z=1)
        c2 = cu(z=2)
        c3 = cu(n="b", z=1)
        c4 = cu(n="b", z=2)
        input_cubes = [c1, c2, c3, c4]
        constraints = [
            iris.Constraint("a") & iris.Constraint(height=1),
            iris.Constraint(height=2),
        ]
        expected_results = {
            "load": [c1, c2, c4],
            "load_cube": "ValueError.*only a single constraint is allowed",
            "load_cubes": r"ConstraintMismatchError.*-> \d+ cubes",
            "load_raw": [c1, c2, c4],
        }
        result = run_testcase(input_cubes, loadfunc_name, constraints=constraints)
        debug_result(result)
        check_result(input_cubes, loadfunc_name, result, expected_results)
