# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the coordinate categorisation functions.
"""

import warnings

import cf_units
import numpy as np
import pytest

import iris
import iris.coord_categorisation as ccat
import iris.coords
import iris.cube
import iris.exceptions
from iris.tests import IrisTest


@pytest.fixture(
    scope="module",
    params=(
        ccat.add_day_of_month,
        ccat.add_day_of_year,
        ccat.add_weekday,
        ccat.add_weekday_fullname,
        ccat.add_weekday_number,
        ccat.add_month,
        ccat.add_month_fullname,
        ccat.add_month_number,
        ccat.add_year,
        ccat.add_season,
        ccat.add_season_number,
        ccat.add_season_year,
        ccat.add_season_membership,
    ),
)
def categorisation_func(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=(
        ccat.add_season,
        ccat.add_season_number,
        ccat.add_season_year,
    ),
)
def season_cat_func(request):
    return request.param


@pytest.fixture(scope="module")
def day_numbers():
    # make a series of 'day numbers' for the time, that slide across month
    # boundaries
    return np.arange(0, 600, 27, dtype=np.int32)


@pytest.fixture
def time_coord(day_numbers):
    return iris.coords.DimCoord(
        day_numbers,
        standard_name="time",
        units=cf_units.Unit("days since epoch", "standard"),
    )


@pytest.fixture
def cube(day_numbers, time_coord):
    _cube = iris.cube.Cube(day_numbers, long_name="test cube", units="metres")
    # use day numbers as data values also (don't actually use this for
    # anything)
    _cube.data = day_numbers
    _cube.add_dim_coord(time_coord, 0)
    return _cube


def test_bad_coord(cube, categorisation_func):
    kwargs = {"name": "my_category"}
    if categorisation_func is ccat.add_season_membership:
        kwargs["season"] = "djf"
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        categorisation_func(cube, "DOES NOT EXIST", **kwargs)


def test_explicit_result_names(cube, categorisation_func):
    result_name = "my_category"
    fmt = "Missing/incorrectly named result for {0!r}"
    # Specify source coordinate by name
    new_cube = cube.copy()
    kwargs = {"name": result_name}
    if categorisation_func is ccat.add_season_membership:
        kwargs["season"] = "djf"
    with warnings.catch_warnings(record=True):
        categorisation_func(new_cube, "time", **kwargs)
    result_coords = new_cube.coords(result_name)
    assert len(result_coords) == 1, fmt.format(categorisation_func.__name__)
    # Specify source coordinate by coordinate reference
    new_cube = cube.copy()
    time = new_cube.coord("time")
    with warnings.catch_warnings(record=True):
        categorisation_func(new_cube, time, **kwargs)
    result_coords = new_cube.coords(result_name)
    assert len(result_coords) == 1, fmt.format(categorisation_func.__name__)


def test_basic(cube, time_coord):
    ccat.add_year(cube, time_coord, "my_year")
    ccat.add_day_of_month(cube, time_coord, "my_day_of_month")
    ccat.add_day_of_year(cube, time_coord, "my_day_of_year")

    ccat.add_month(cube, time_coord, "my_month")
    ccat.add_month_fullname(cube, time_coord, "my_month_fullname")
    ccat.add_month_number(cube, time_coord, "my_month_number")

    ccat.add_weekday(cube, time_coord, "my_weekday")
    ccat.add_weekday_number(cube, time_coord, "my_weekday_number")
    ccat.add_weekday_fullname(cube, time_coord, "my_weekday_fullname")

    ccat.add_season(cube, time_coord, "my_season")
    ccat.add_season_number(cube, time_coord, "my_season_number")
    ccat.add_season_year(cube, time_coord, "my_season_year")

    # also test 'generic' categorisation interface
    def _month_in_quarter(coord, pt_value):
        date = coord.units.num2date(pt_value)
        return (date.month - 1) % 3

    ccat.add_categorised_coord(
        cube, "my_month_in_quarter", time_coord, _month_in_quarter
    )

    # To ensure consistent results between 32-bit and 64-bit
    # platforms, ensure all the numeric categorisation coordinates
    # are always stored as int64.
    for coord in cube.coords():
        if coord.long_name is not None and coord.points.dtype.kind == "i":
            coord.points = coord.points.astype(np.int64)

    # check values
    IrisTest.assertCML(IrisTest(), cube, ("categorisation", "quickcheck.cml"))


def test_add_season_nonstandard(cube, time_coord):
    # season categorisations work for non-standard seasons?
    seasons = ["djfm", "amjj", "ason"]
    ccat.add_season(cube, time_coord, name="seasons", seasons=seasons)
    ccat.add_season_number(cube, time_coord, name="season_numbers", seasons=seasons)
    ccat.add_season_year(cube, time_coord, name="season_years", seasons=seasons)
    IrisTest.assertCML(IrisTest(), cube, ("categorisation", "customcheck.cml"))


@pytest.mark.parametrize("backwards", [None, False, True])
@pytest.mark.parametrize(
    "nonstandard",
    [False, True],
    ids=["standard_seasons", "nonstandard_seasons"],
)
def test_add_season_year(cube, time_coord, backwards, nonstandard):
    """Specific test to account for the extra use_year_at_season_start argument."""
    kwargs = dict(
        cube=cube,
        coord=time_coord,
        name="season_years",
        use_year_at_season_start=backwards,
    )
    if nonstandard:
        kwargs["seasons"] = ["ndjfm", "amjj", "aso"]

    # Based on the actual years of each date.
    expected_years = np.array(([1970] * 14) + ([1971] * 9))
    # Subset to just the 'season' of interest.
    season_slice = np.s_[12:17]
    expected_years = expected_years[season_slice]

    # Single indices to examine to test the handling of specific months.
    nov = 0
    dec = 1
    jan = 2
    feb = 3
    mar = 4

    # Set the expected deviations from the actual date years.
    if backwards is True:
        expected_years[jan] = 1970
        expected_years[feb] = 1970
        if nonstandard:
            expected_years[mar] = 1970
    else:
        # Either False or None - False being the default behaviour.
        expected_years[dec] = 1971
        if nonstandard:
            expected_years[nov] = 1971

    ccat.add_season_year(**kwargs)
    actual_years = cube.coord(kwargs["name"]).points
    # Subset to just the 'season' of interest.
    actual_years = actual_years[season_slice]

    np.testing.assert_array_almost_equal(actual_years, expected_years)


def test_add_season_membership(cube):
    # season membership identifies correct seasons?
    season = "djf"
    ccat.add_season_membership(cube, "time", season, name="in_season")
    ccat.add_season(cube, "time")
    coord_season = cube.coord("season")
    coord_membership = cube.coord("in_season")
    season_locations = np.where(coord_season.points == season)[0]
    membership_locations = np.where(coord_membership.points)[0]
    np.testing.assert_array_almost_equal(membership_locations, season_locations)


def test_add_season_invalid_spec(cube, season_cat_func):
    # custom seasons with an invalid season raises an error?
    seasons = ("djf", "maj", "jja", "son")  # MAJ not a season!
    with pytest.raises(ValueError):
        season_cat_func(cube, "time", name="my_category", seasons=seasons)


def test_add_season_repeated_months(cube, season_cat_func):
    # custom seasons with repeated months raises an error?
    seasons = ("djfm", "mam", "jja", "son")
    with pytest.raises(ValueError):
        season_cat_func(cube, "time", name="my_category", seasons=seasons)


def test_add_season_missing_months(cube, season_cat_func):
    # custom seasons with missing months raises an error?
    seasons = ("djfm", "amjj")
    with pytest.raises(ValueError):
        season_cat_func(cube, "time", name="my_category", seasons=seasons)


def test_add_season_membership_invalid_spec(cube):
    season = "maj"  # not a season!
    with pytest.raises(ValueError):
        ccat.add_season_membership(cube, "time", season, name="maj_season")
