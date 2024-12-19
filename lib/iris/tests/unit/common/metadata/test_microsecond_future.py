# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the opt-in FUTURE.date_microseconds behaviour."""

import warnings

import cf_units
import numpy as np
from numpy.testing import assert_array_equal
from packaging.version import Version
import pytest

from iris import FUTURE
from iris.coords import DimCoord

cf_units_legacy = Version(cf_units.__version__) < Version("3.3.0")


@pytest.fixture(
    params=[0, 1000, 500000],
    ids=["no_microseconds", "1_millisecond", "half_second"],
)
def time_coord(request) -> tuple[bool, DimCoord]:
    points = np.array([0.0, 1.0, 2.0])
    points += request.param / 1e6
    return request.param, DimCoord(
        points,
        "time",
        units="seconds since 1970-01-01 00:00:00",
    )


@pytest.fixture(
    params=[False, True],
    ids=["without_future", "with_future"],
)
def future_date_microseconds(request):
    FUTURE.date_microseconds = request.param
    yield request.param
    FUTURE.date_microseconds = False


def test_warning(time_coord, future_date_microseconds):
    # Warning should be raised whether the coordinate has microseconds or not.
    #  Want users to be aware, and opt-in, as early as possible.
    n_microseconds, coord = time_coord

    def _op():
        _ = coord.units.num2date(coord.points)

    if future_date_microseconds:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            _op()
    else:
        with pytest.warns(FutureWarning):
            _op()


@pytest.mark.parametrize(
    "indexing",
    (np.s_[0], np.s_[:], np.s_[:, np.newaxis]),
    ids=("single", "array", "array_2d"),
)
def test_num2date(time_coord, future_date_microseconds, indexing):
    n_microseconds, coord = time_coord
    result = coord.units.num2date(coord.points[indexing])

    if indexing == np.s_[0]:
        assert hasattr(result, "microsecond")
        # Convert to iterable for more consistency downstream.
        result = [result]
    else:
        assert hasattr(result, "shape")
        assert hasattr(result.flatten()[0], "microsecond")
        result = result.flatten()

    expected_microseconds = n_microseconds
    if not future_date_microseconds or cf_units_legacy:
        expected_microseconds = 0

    result_microseconds = np.array([r.microsecond for r in result])
    assert_array_equal(result_microseconds, expected_microseconds)


def test_roundup(time_coord, future_date_microseconds):
    n_microseconds, coord = time_coord
    result = coord.units.num2date(coord.points)

    expected_seconds = np.floor(coord.points)
    if n_microseconds >= 500000 and (not future_date_microseconds or cf_units_legacy):
        # Legacy cf-units versions round microseconds and ignore the future flag.
        expected_seconds += 1

    result_seconds = np.array([r.second for r in result])
    assert_array_equal(result_seconds, expected_seconds)
