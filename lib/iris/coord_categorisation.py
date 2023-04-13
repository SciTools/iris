# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Cube functions for coordinate categorisation.

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

import numpy as np

import iris.coords


def add_categorised_coord(
    cube, name, from_coord, category_function, units="1"
):
    """
    Add a new coordinate to a cube, by categorising an existing one.

    Make a new :class:`iris.coords.AuxCoord` from mapped values, and add
    it to the cube.

    Args:

    * cube (:class:`iris.cube.Cube`):
        the cube containing 'from_coord'.  The new coord will be added into it.
    * name (string):
        name of the created coordinate
    * from_coord (:class:`iris.coords.Coord` or string):
        coordinate in 'cube', or the name of one
    * category_function (callable):
        function(coordinate, value), returning a category value for a
        coordinate point-value

    Kwargs:

    * units:
        units of the category value, typically 'no_unit' or '1'.
    """
    # Interpret coord, if given as a name
    if isinstance(from_coord, str):
        from_coord = cube.coord(from_coord)

    if len(cube.coords(name)) > 0:
        msg = 'A coordinate "%s" already exists in the cube.' % name
        raise ValueError(msg)

    # Construct new coordinate by mapping values, using numpy.vectorize to
    # support multi-dimensional coords.
    # Test whether the result contains strings. If it does we must manually
    # force the dtype because of a numpy bug (see numpy #3270 on GitHub).
    result = category_function(from_coord, from_coord.points.ravel()[0])
    if isinstance(result, str):
        str_vectorised_fn = np.vectorize(category_function, otypes=[object])

        def vectorised_fn(*args):
            # Use a common type for string arrays (N.B. limited to 64 chars).
            return str_vectorised_fn(*args).astype("|U64")

    else:
        vectorised_fn = np.vectorize(category_function)
    new_coord = iris.coords.AuxCoord(
        vectorised_fn(from_coord, from_coord.points),
        units=units,
        attributes=from_coord.attributes.copy(),
    )
    new_coord.rename(name)

    # Add into the cube
    cube.add_aux_coord(new_coord, cube.coord_dims(from_coord))


# ======================================
# Specific functions for particular purposes
#
# NOTE: all the existing ones are calendar operations, so are for 'Time'
# coordinates only
#


# Private "helper" function
def _pt_date(coord, time):
    """
    Return the datetime of a time-coordinate point.

    Args:

    * coord (Coord):
        coordinate (must be Time-type)
    * time (float):
        value of a coordinate point

    Returns:
        cftime.datetime

    """
    # NOTE: All of the currently defined categorisation functions are
    # calendar operations on Time coordinates.
    return coord.units.num2date(time, only_use_cftime_datetimes=True)


# --------------------------------------------
# Time categorisations : calendar date components


def add_year(cube, coord, name="year"):
    """Add a categorical calendar-year coordinate."""
    add_categorised_coord(
        cube, name, coord, lambda coord, x: _pt_date(coord, x).year
    )


def add_month_number(cube, coord, name="month_number"):
    """Add a categorical month coordinate, values 1..12."""
    add_categorised_coord(
        cube, name, coord, lambda coord, x: _pt_date(coord, x).month
    )


def add_month_fullname(cube, coord, name="month_fullname"):
    """Add a categorical month coordinate, values 'January'..'December'."""
    add_categorised_coord(
        cube,
        name,
        coord,
        lambda coord, x: calendar.month_name[_pt_date(coord, x).month],
        units="no_unit",
    )


def add_month(cube, coord, name="month"):
    """Add a categorical month coordinate, values 'Jan'..'Dec'."""
    add_categorised_coord(
        cube,
        name,
        coord,
        lambda coord, x: calendar.month_abbr[_pt_date(coord, x).month],
        units="no_unit",
    )


def add_day_of_month(cube, coord, name="day_of_month"):
    """Add a categorical day-of-month coordinate, values 1..31."""
    add_categorised_coord(
        cube, name, coord, lambda coord, x: _pt_date(coord, x).day
    )


def add_day_of_year(cube, coord, name="day_of_year"):
    """
    Add a categorical day-of-year coordinate, values 1..365
    (1..366 in leap years).

    """
    # Note: cftime.datetime objects return a normal tuple from timetuple(),
    # unlike datetime.datetime objects that return a namedtuple.
    # Index the time tuple (element 7 is day of year) instead of using named
    # element tm_yday.
    add_categorised_coord(
        cube, name, coord, lambda coord, x: _pt_date(coord, x).timetuple()[7]
    )


# --------------------------------------------
# Time categorisations : days of the week


def add_weekday_number(cube, coord, name="weekday_number"):
    """Add a categorical weekday coordinate, values 0..6  [0=Monday]."""
    add_categorised_coord(
        cube, name, coord, lambda coord, x: _pt_date(coord, x).dayofwk
    )


def add_weekday_fullname(cube, coord, name="weekday_fullname"):
    """Add a categorical weekday coordinate, values 'Monday'..'Sunday'."""
    add_categorised_coord(
        cube,
        name,
        coord,
        lambda coord, x: calendar.day_name[_pt_date(coord, x).dayofwk],
        units="no_unit",
    )


def add_weekday(cube, coord, name="weekday"):
    """Add a categorical weekday coordinate, values 'Mon'..'Sun'."""
    add_categorised_coord(
        cube,
        name,
        coord,
        lambda coord, x: calendar.day_abbr[_pt_date(coord, x).dayofwk],
        units="no_unit",
    )


# --------------------------------------------
# Time categorisations : hour of the day


def add_hour(cube, coord, name="hour"):
    """Add a categorical hour coordinate, values 0..23."""
    add_categorised_coord(
        cube, name, coord, lambda coord, x: _pt_date(coord, x).hour
    )


# ----------------------------------------------
# Time categorisations : meteorological seasons


def _months_in_season(season):
    """
    Returns a list of month numbers corresponding to each month in the
    given season.

    """
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
            "some months do not appear in any season: "
            "{!s}".format(", ".join(not_present))
        )
    # Make a list of months that appear multiple times...
    multi_present = [
        calendar.month_abbr[month] for month in range(1, 13) if c[month] > 1
    ]
    if multi_present:
        raise ValueError(
            "some months appear in more than one season: "
            "{!s}".format(", ".join(multi_present))
        )
    return


def _month_year_adjusts(seasons):
    """Compute the year adjustments required for each month.

    These determine whether the month belongs to a season in the same
    year or is in the start of a season that counts towards the next
    year.

    """
    month_year_adjusts = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for season in seasons:
        months = _months_in_season(season)
        for month in months:
            if month > months[-1]:
                month_year_adjusts[month] = 1
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


def add_season(
    cube, coord, name="season", seasons=("djf", "mam", "jja", "son")
):
    """
    Add a categorical season-of-year coordinate, with user specified
    seasons.

    Args:

    * cube (:class:`iris.cube.Cube`):
        The cube containing 'coord'. The new coord will be added into
        it.
    * coord (:class:`iris.coords.Coord` or string):
        Coordinate in 'cube', or its name, representing time.

    Kwargs:

    * name (string):
        Name of the created coordinate. Defaults to "season".
    * seasons (:class:`list` of strings):
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
    def _season(coord, value):
        dt = _pt_date(coord, value)
        return seasons[month_season_numbers[dt.month]]

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season, units="no_unit")


def add_season_number(
    cube, coord, name="season_number", seasons=("djf", "mam", "jja", "son")
):
    """
    Add a categorical season-of-year coordinate, values 0..N-1 where
    N is the number of user specified seasons.

    Args:

    * cube (:class:`iris.cube.Cube`):
        The cube containing 'coord'. The new coord will be added into
        it.
    * coord (:class:`iris.coords.Coord` or string):
        Coordinate in 'cube', or its name, representing time.

    Kwargs:

    * name (string):
        Name of the created coordinate. Defaults to "season_number".
    * seasons (:class:`list` of strings):
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
    def _season_number(coord, value):
        dt = _pt_date(coord, value)
        return month_season_numbers[dt.month]

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season_number)


def add_season_year(
    cube, coord, name="season_year", seasons=("djf", "mam", "jja", "son")
):
    """
    Add a categorical year-of-season coordinate, with user specified
    seasons.

    Args:

    * cube (:class:`iris.cube.Cube`):
        The cube containing 'coord'. The new coord will be added into
        it.
    * coord (:class:`iris.coords.Coord` or string):
        Coordinate in 'cube', or its name, representing time.

    Kwargs:

    * name (string):
        Name of the created coordinate. Defaults to "season_year".
    * seasons (:class:`list` of strings):
        List of seasons defined by month abbreviations. Each month must
        appear once and only once. Defaults to standard meteorological
        seasons ('djf', 'mam', 'jja', 'son').

    """
    # Check that the seasons are valid.
    _validate_seasons(seasons)
    # Define the adjustments to be made to the year.
    month_year_adjusts = _month_year_adjusts(seasons)

    # Define a categorisation function.
    def _season_year(coord, value):
        dt = _pt_date(coord, value)
        year = dt.year
        year += month_year_adjusts[dt.month]
        return year

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season_year)


def add_season_membership(cube, coord, season, name="season_membership"):
    """
    Add a categorical season membership coordinate for a user specified
    season.

    The coordinate has the value True for every time that is within the
    given season, and the value False otherwise.

    Args:

    * cube (:class:`iris.cube.Cube`):
        The cube containing 'coord'. The new coord will be added into
        it.
    * coord (:class:`iris.coords.Coord` or string):
        Coordinate in 'cube', or its name, representing time.
    * season (string):
        Season defined by month abbreviations.

    Kwargs:

    * name (string):
        Name of the created coordinate. Defaults to "season_membership".

    """
    months = _months_in_season(season)

    def _season_membership(coord, value):
        dt = _pt_date(coord, value)
        if dt.month in months:
            return True
        return False

    add_categorised_coord(cube, name, coord, _season_membership)
