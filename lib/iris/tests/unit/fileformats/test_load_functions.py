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
from iris.coords import DimCoord
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


def check_result(input_cubes, loadfunc_name, result, expected_results):
    if loadfunc_name == "load_raw":
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
            "load_cubes": "ConstraintMismatchError.*-> 2 cubes",
        }
        result = run_testcase(input_cubes, loadfunc_name)
        check_result(input_cubes, loadfunc_name, result, expected_results)
