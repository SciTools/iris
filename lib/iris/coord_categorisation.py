# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Cube functions for coordinate categorisation.

All the functions provided here add a new coordinate to a cube.

* The function :func:`add_categorised_coord` performs a generic
  coordinate categorisation.
* The other functions all implement specific common cases
  (e.g. :func:`add_day_of_month`).
  Currently, these are all calendar functions, so they only apply to
  "Time coordinates".

"""

import calendar
import collections
import inspect
from typing import Callable

import cftime
import numpy as np

import iris.coords
import iris.cube


def add_categorised_coord(
    cube: iris.cube.Cube,
    name: str,
    from_coord: iris.coords.DimCoord | iris.coords.AuxCoord | str,
    category_function: Callable,
    units: str = "1",
) -> None:
    """Add a new coordinate to a cube, by categorising an existing one.

    Make a new :class:`iris.coords.AuxCoord` from mapped values, and add
    it to the cube.

    Parameters
    ----------
    cube :
        The cube containing 'from_coord'. The new coord will be added into it.
    name :
        Name of the created coordinate.
    from_coord :
        Coordinate in 'cube', or the name of one.
    category_function :
        Function(coordinate, value), returning a category value for a coordinate
        point-value. If ``value`` has a type hint :obj:`cftime.datetime`, the
        coordinate points are translated to :obj:`cftime.datetime` s before
        calling ``category_function``.
    units :
        Units of the category value, typically 'no_unit' or '1'.
    """
    # Interpret coord, if given as a name
    coord = cube.coord(from_coord) if isinstance(from_coord, str) else from_coord

    if len(cube.coords(name)) > 0:
        msg = 'A coordinate "%s" already exists in the cube.' % name
        raise ValueError(msg)

    # Translate the coordinate points to cftime datetimes if requested.
    value_param = list(inspect.signature(category_function).parameters.values())[1]
    if issubclass(value_param.annotation, cftime.datetime):
        points = coord.units.num2date(coord.points, only_use_cftime_datetimes=True)
    else:
        points = coord.points

    # Construct new coordinate by mapping values, using numpy.vectorize to
    # support multi-dimensional coords.
    # Test whether the result contains strings. If it does we must manually
    # force the dtype because of a numpy bug (see numpy #3270 on GitHub).
    result = category_function(coord, points.ravel()[0])
    if isinstance(result, str):
        str_vectorised_fn = np.vectorize(category_function, otypes=[object])

        def vectorised_fn(*args):
            # Use a common type for string arrays (N.B. limited to 64 chars).
            return str_vectorised_fn(*args).astype("|U64")

    else:
        vectorised_fn = np.vectorize(category_function)
    new_coord = iris.coords.AuxCoord(
        vectorised_fn(coord, points),
        units=units,
        attributes=coord.attributes.copy(),
    )
    new_coord.rename(name)

    # Add into the cube
    cube.add_aux_coord(new_coord, cube.coord_dims(coord))


# ======================================
# Specific functions for particular purposes
#
# NOTE: all the existing ones are calendar operations, so are for 'Time'
# coordinates only
#

# --------------------------------------------
# Time categorisations : calendar date components


def add_year(cube, coord, name="year"):
    """Add a categorical calendar-year coordinate."""

    def get_year(_, value: cftime.datetime) -> int:
        return value.year

    add_categorised_coord(cube, name, coord, get_year)


def add_month_number(cube, coord, name="month_number"):
    """Add a categorical month coordinate, values 1..12."""

    def get_month_number(_, value: cftime.datetime) -> int:
        return value.month

    add_categorised_coord(cube, name, coord, get_month_number)


def add_month_fullname(cube, coord, name="month_fullname"):
    """Add a categorical month coordinate, values 'January'..'December'."""

    def get_month_fullname(_, value: cftime.datetime) -> str:
        return calendar.month_name[value.month]

    add_categorised_coord(cube, name, coord, get_month_fullname, units="no_unit")


def add_month(cube, coord, name="month"):
    """Add a categorical month coordinate, values 'Jan'..'Dec'."""

    def get_month_abbr(_, value: cftime.datetime) -> str:
        return calendar.month_abbr[value.month]

    add_categorised_coord(cube, name, coord, get_month_abbr, units="no_unit")


def add_day_of_month(cube, coord, name="day_of_month"):
    """Add a categorical day-of-month coordinate, values 1..31."""

    def get_day_of_month(_, value: cftime.datetime) -> int:
        return value.day

    add_categorised_coord(cube, name, coord, get_day_of_month)


def add_day_of_year(cube, coord, name="day_of_year"):
    """Add a categorical day-of-year coordinate, values 1..365 (1..366 in leap years)."""

    def get_day_of_year(_, value: cftime.datetime) -> int:
        return value.timetuple().tm_yday

    add_categorised_coord(cube, name, coord, get_day_of_year)


# --------------------------------------------
# Time categorisations : days of the week


def add_weekday_number(cube, coord, name="weekday_number"):
    """Add a categorical weekday coordinate, values 0..6  [0=Monday]."""

    def get_weekday_number(_, value: cftime.datetime) -> int:
        return value.dayofwk

    add_categorised_coord(cube, name, coord, get_weekday_number)


def add_weekday_fullname(cube, coord, name="weekday_fullname"):
    """Add a categorical weekday coordinate, values 'Monday'..'Sunday'."""

    def get_weekday_fullname(_, value: cftime.datetime) -> str:
        return calendar.day_name[value.dayofwk]

    add_categorised_coord(cube, name, coord, get_weekday_fullname, units="no_unit")


def add_weekday(cube, coord, name="weekday"):
    """Add a categorical weekday coordinate, values 'Mon'..'Sun'."""

    def get_weekday(_, value: cftime.datetime) -> str:
        return calendar.day_abbr[value.dayofwk]

    add_categorised_coord(cube, name, coord, get_weekday, units="no_unit")


# --------------------------------------------
# Time categorisations : hour of the day


def add_hour(cube, coord, name="hour"):
    """Add a categorical hour coordinate, values 0..23."""

    def get_hour(_, value: cftime.datetime) -> int:
        return value.hour

    add_categorised_coord(cube, name, coord, get_hour)


# ----------------------------------------------
# Time categorisations : meteorological seasons


def _months_in_season(season):
    """Return a list of month numbers corresponding to each month in the given season."""
    cyclic_months = "jfmamjjasondjfmamjjasond"
    m0 = cyclic_months.find(season.lower())
    if m0 < 0:
        # Can't match the season, raise an error.
        raise ValueError("unrecognised season: {!s}".format(season))
    m1 = m0 + len(season)
    return [(month % 12) + 1 for month in range(m0, m1)]


def _validate_seasons(seasons):
    """Check that a set of seasons is valid.

    Validity means that all months are included in a season, and no
    month is assigned to more than one season.

    Raises ValueError if either of the conditions is not met, returns
    None otherwise.

    """
    c = collections.Counter()
    for season in seasons:
        c.update(_months_in_season(season))
    # Make a list of months that are not present...
    not_present = [
        calendar.month_abbr[month] for month in range(1, 13) if month not in c
    ]
    if not_present:
        raise ValueError(
            "some months do not appear in any season: {!s}".format(
                ", ".join(not_present)
            )
        )
    # Make a list of months that appear multiple times...
    multi_present = [
        calendar.month_abbr[month] for month in range(1, 13) if c[month] > 1
    ]
    if multi_present:
        raise ValueError(
            "some months appear in more than one season: {!s}".format(
                ", ".join(multi_present)
            )
        )
    return


def _month_year_adjusts(seasons, use_year_at_season_start=False):
    """Compute the year adjustments required for each month.

    These adjustments ensure that no season spans two years by assigning months
    to the **next** year (use_year_at_season_start is False) or the
    **previous** year (use_year_at_season_start is True). E.g. Winter - djf:
    either assign Dec to the next year, or Jan and Feb to the previous year.

    """
    # 1 'slot' for each month, with an extra leading 'slot' because months
    #  are 1-indexed - January is 1, therefore corresponding to the 2nd
    #  array index.
    month_year_adjusts = np.zeros(13, dtype=int)

    for season in seasons:
        months = np.array(_months_in_season(season))
        if use_year_at_season_start:
            months_to_shift = months < months[0]
            year_shift = -1
        else:
            # Sending forwards.
            months_to_shift = months > months[-1]
            year_shift = 1
        indices_to_shift = months[np.flatnonzero(months_to_shift)]
        month_year_adjusts[indices_to_shift] = year_shift

    return month_year_adjusts


def _month_season_numbers(seasons):
    """Compute a mapping between months and season number.

    Returns a list to be indexed by month number, where the value at
    each index is the number of the season that month belongs to.

    """
    month_season_numbers = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for season_number, season in enumerate(seasons):
        for month in _months_in_season(season):
            month_season_numbers[month] = season_number
    return month_season_numbers


def add_season(cube, coord, name="season", seasons=("djf", "mam", "jja", "son")):
    """Add a categorical season-of-year coordinate, with user specified seasons.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube containing 'coord'. The new coord will be added into
        it.
    coord : :class:`iris.coords.Coord` or str
        Coordinate in 'cube', or its name, representing time.
    name : str, default="season"
        Name of the created coordinate. Defaults to "season".
    seasons : :class:`list` of str, optional
        List of seasons defined by month abbreviations. Each month must
        appear once and only once. Defaults to standard meteorological
        seasons ('djf', 'mam', 'jja', 'son').

    """
    # Check that the seasons are valid.
    _validate_seasons(seasons)
    # Get a list of the season number each month is is, using month numbers
    # as the indices.
    month_season_numbers = _month_season_numbers(seasons)

    # Define a categorisation function.
    def _season(_, value: cftime.datetime) -> str:
        return seasons[month_season_numbers[value.month]]

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season, units="no_unit")


def add_season_number(
    cube, coord, name="season_number", seasons=("djf", "mam", "jja", "son")
):
    """Add a categorical season-of-year coordinate.

    Add a categorical season-of-year coordinate, values 0..N-1 where
    N is the number of user specified seasons.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube containing 'coord'. The new coord will be added into
        it.
    coord : :class:`iris.coords.Coord` or str
        Coordinate in 'cube', or its name, representing time.
    name : str, default="season"
        Name of the created coordinate. Defaults to "season_number".
    seasons : :class:`list` of str, optional
        List of seasons defined by month abbreviations. Each month must
        appear once and only once. Defaults to standard meteorological
        seasons ('djf', 'mam', 'jja', 'son').

    """
    # Check that the seasons are valid.
    _validate_seasons(seasons)
    # Get a list of the season number each month is is, using month numbers
    # as the indices.
    month_season_numbers = _month_season_numbers(seasons)

    # Define a categorisation function.
    def _season_number(_, value: cftime.datetime) -> int:
        return month_season_numbers[value.month]

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season_number)


def add_season_year(
    cube,
    coord,
    name="season_year",
    seasons=("djf", "mam", "jja", "son"),
    use_year_at_season_start=False,
):
    """Add a categorical year-of-season coordinate, with user specified seasons.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube containing `coord`. The new coord will be added into it.
    coord : :class:`iris.coords.Coord` or str
        Coordinate in `cube`, or its name, representing time.
    name : str, default="season_year"
        Name of the created coordinate.
    seasons : tuple of str, default=("djf", "mam", "jja", "son")
        List of seasons defined by month abbreviations. Each month must
        appear once and only once. Defaults to standard meteorological
        seasons (``djf``, ``mam``, ``jja``, ``son``).
    use_year_at_season_start : bool, default=False
        Seasons spanning the year boundary (e.g. Winter ``djf``) will belong
        fully to the following year by default (e.g. the year of Jan and Feb).
        Set to ``True`` for spanning seasons to belong to the preceding
        year (e.g. the year of Dec) instead.

    """
    # Check that the seasons are valid.
    _validate_seasons(seasons)
    # Define the adjustments to be made to the year.
    month_year_adjusts = _month_year_adjusts(
        seasons, use_year_at_season_start=use_year_at_season_start
    )

    # Define a categorisation function.
    def _season_year(_, value: cftime.datetime) -> int:
        year = value.year
        year += month_year_adjusts[value.month]
        return year

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season_year)


def add_season_membership(cube, coord, season, name="season_membership"):
    """Add a categorical season membership coordinate for a user specified season.

    The coordinate has the value True for every time that is within the
    given season, and the value False otherwise.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
        The cube containing 'coord'. The new coord will be added into
        it.
    coord : :class:`iris.coords.Coord` or str
        Coordinate in 'cube', or its name, representing time.
    season : str
        Season defined by month abbreviations.
    name : str, default="season_membership"
        Name of the created coordinate. Defaults to "season_membership".

    """
    months = _months_in_season(season)

    def _season_membership(_, value: cftime.datetime) -> bool:
        return value.month in months

    add_categorised_coord(cube, name, coord, _season_membership)
