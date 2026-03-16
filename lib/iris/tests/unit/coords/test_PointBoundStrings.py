# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coords.PointBoundString class."""

from cf_units import Unit
import dask.array as da
import numpy as np

from iris._lazy_data import is_lazy_data
from iris.coords import AuxCoord, PointBoundStrings


def test_PointBoundStrings_lazy():
    lazy_points = da.arange(5, dtype=np.float64)
    lazy_bounds = da.arange(10, dtype=np.float64).reshape([5, 2])

    lazy_coord = AuxCoord(lazy_points, bounds=lazy_bounds, standard_name="latitude")

    fmt = ".0f"
    pbs = lazy_coord.as_string_arrays(fmt)
    assert is_lazy_data(pbs._core_bounds)
    assert pbs._bounds is None

    expected_bounds = "[['0' '1']\n ['2' '3']\n ['4' '5']\n ['6' '7']\n ['8' '9']]"
    assert np.array2string(pbs.bounds) == expected_bounds
    assert pbs._core_bounds is None
    assert pbs._bounds is pbs.bounds
    assert is_lazy_data(pbs._core_points)
    assert pbs._points is None

    assert lazy_coord.has_lazy_points()
    assert lazy_coord.has_lazy_bounds()


def test_PointBoundStrings_no_bounds():
    points = np.arange(5, dtype=np.float64)

    coord = AuxCoord(points, standard_name="latitude")
    pbs = coord.as_string_arrays()

    expected_output = "Points:\n['0.0' '1.0' '2.0' '3.0' '4.0']"
    assert str(pbs) == expected_output

    expected_points = np.array(['0.0', '1.0', '2.0', '3.0', '4.0'])
    assert np.array_equal(pbs.points, expected_points)

    assert pbs.bounds is None


def test_PointBoundStrings_time_coord():
    time_unit = Unit("days since epoch")
    points = np.arange(5)
    bounds = np.arange(10).reshape([5, 2])

    pbs_unformatted = PointBoundStrings(points, bounds, time_unit)
    expected_unformatted = (
        "Points:\n"
        "['1970-01-01 00:00:00' '1970-01-02 00:00:00' '1970-01-03 00:00:00'\n"
        " '1970-01-04 00:00:00' '1970-01-05 00:00:00']\n"
        "Bounds:\n"
        "[['1970-01-01 00:00:00' '1970-01-02 00:00:00']\n"
        " ['1970-01-03 00:00:00' '1970-01-04 00:00:00']\n"
        " ['1970-01-05 00:00:00' '1970-01-06 00:00:00']\n"
        " ['1970-01-07 00:00:00' '1970-01-08 00:00:00']\n"
        " ['1970-01-09 00:00:00' '1970-01-10 00:00:00']]"
    )
    assert str(pbs_unformatted) == expected_unformatted
    fmt = "%Y-%m-%d"
    pbs_formatted = PointBoundStrings(points, bounds, time_unit, fmt=fmt)
    expected_formatted = (
        "Points:\n"
        "['1970-01-01' '1970-01-02' '1970-01-03' '1970-01-04' '1970-01-05']\n"
        "Bounds:\n"
        "[['1970-01-01' '1970-01-02']\n"
        " ['1970-01-03' '1970-01-04']\n"
        " ['1970-01-05' '1970-01-06']\n"
        " ['1970-01-07' '1970-01-08']\n"
        " ['1970-01-09' '1970-01-10']]"
    )
    assert str(pbs_formatted) == expected_formatted
