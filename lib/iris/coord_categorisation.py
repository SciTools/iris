# (C) British Crown Copyright 2010 - 2019, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import calendar
import collections

import numpy as np

import iris.coords


def add_categorised_coord(cube, name, from_coord, category_function,
                          units='1', vectorize=True):
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
    * vectorize (bool):
        Whether the given ``category_function`` should be vectorized with
        ``np.vectorize`` before it is called on the coordinate points.
        Providing an optimized, and pre-vectorizedm, function that operates
        on all coordinate values at the same time can have a significant
        performance improvement when categorising large coordinates.
        Default: False.

    """
    # Interpret coord, if given as a name
    if isinstance(from_coord, six.string_types):
        from_coord = cube.coord(from_coord)

    if len(cube.coords(name)) > 0:
        msg = 'A coordinate "%s" already exists in the cube.' % name
        raise ValueError(msg)

    if vectorize:
        vectorised_fn = np.vectorize(category_function)
    else:
        vectorised_fn = category_function
    new_coord = iris.coords.AuxCoord(
        vectorised_fn(from_coord, from_coord.points),
        units=units, attributes=from_coord.attributes.copy())
    new_coord.rename(name)

    # Add into the cube
    cube.add_aux_coord(new_coord, cube.coord_dims(from_coord))


# --------------------------------------------
# Time categorisations : calendar date components

def add_year(cube, coord, name='year'):
    """Add a categorical calendar-year coordinate."""
    year = np.vectorize(lambda dt: dt.year)
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: year(coord.units.num2date(xs)),
        vectorize=False)


def add_month_number(cube, coord, name='month_number'):
    """Add a categorical month coordinate, values 1..12."""
    month = np.vectorize(lambda dt: dt.month)
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: month(coord.units.num2date(xs)),
        vectorize=False)


def add_month_fullname(cube, coord, name='month_fullname'):
    """Add a categorical month coordinate, values 'January'..'December'."""
    month_name = np.vectorize(lambda dt: calendar.month_name[dt.month])
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: month_name(coord.units.num2date(xs)),
        units='no_unit', vectorize=False)


def add_month(cube, coord, name='month'):
    """Add a categorical month coordinate, values 'Jan'..'Dec'."""
    month_abbr = np.vectorize(lambda dt: calendar.month_abbr[dt.month])
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: month_abbr(coord.units.num2date(xs)),
        units='no_unit', vectorize=False)


def add_day_of_month(cube, coord, name='day_of_month'):
    """Add a categorical day-of-month coordinate, values 1..31."""
    day = np.vectorize(lambda dt: dt.day)
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: day(coord.units.num2date(xs)),
        vectorize=False)


def add_day_of_year(cube, coord, name='day_of_year'):
    """
    Add a categorical day-of-year coordinate, values 1..365
    (1..366 in leap years).

    """
    # Note: cftime.datetime objects return a normal tuple from timetuple(),
    # unlike datetime.datetime objects that return a namedtuple.
    # Index the time tuple (element 7 is day of year) instead of using named
    # element tm_yday.
    day_of_year = np.vectorize(lambda dt: dt.timetuple()[7])
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: day_of_year(coord.units.num2date(xs)),
        vectorize=False)


# --------------------------------------------
# Time categorisations : days of the week

def add_weekday_number(cube, coord, name='weekday_number'):
    """Add a categorical weekday coordinate, values 0..6  [0=Monday]."""
    weekday = np.vectorize(lambda dt: dt.weekday())
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: weekday(coord.units.num2date(xs)),
        vectorize=False)


def add_weekday_fullname(cube, coord, name='weekday_fullname'):
    """Add a categorical weekday coordinate, values 'Monday'..'Sunday'."""
    weekday = np.vectorize(lambda dt: calendar.day_name[dt.weekday()])
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: weekday(coord.units.num2date(xs)),
        units='no_unit', vectorize=False)


def add_weekday(cube, coord, name='weekday'):
    """Add a categorical weekday coordinate, values 'Mon'..'Sun'."""
    weekday = np.vectorize(lambda dt: calendar.day_abbr[dt.weekday()])
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: weekday(coord.units.num2date(xs)),
        units='no_unit', vectorize=False)


# --------------------------------------------
# Time categorisations : hour of the day


def add_hour(cube, coord, name='hour'):
    """Add a categorical hour coordinate, values 0..23."""
    hour = np.vectorize(lambda dt: dt.hour)
    add_categorised_coord(
        cube, name, coord,
        lambda coord, xs: hour(coord.units.num2date(xs)),
        vectorize=False)


# ----------------------------------------------
# Time categorisations : meteorological seasons


def _months_in_season(season):
    """
    Returns a list of month numbers corresponding to each month in the
    given season.

    """
    cyclic_months = 'jfmamjjasondjfmamjjasond'
    m0 = cyclic_months.find(season.lower())
    if m0 < 0:
        # Can't match the season, raise an error.
        raise ValueError('unrecognised season: {!s}'.format(season))
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
    not_present = [calendar.month_abbr[month] for month in range(1, 13)
                   if month not in c]
    if not_present:
        raise ValueError('some months do not appear in any season: '
                         '{!s}'.format(', '.join(not_present)))
    # Make a list of months that appear multiple times...
    multi_present = [calendar.month_abbr[month] for month in range(1, 13)
                     if c[month] > 1]
    if multi_present:
        raise ValueError('some months appear in more than one season: '
                         '{!s}'.format(', '.join(multi_present)))
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


def add_season(cube, coord, name='season',
               seasons=('djf', 'mam', 'jja', 'son')):
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
        dt = coord.units.num2date(value)
        return seasons[month_season_numbers[dt.month]]

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season, units='no_unit')


def add_season_number(cube, coord, name='season_number',
                      seasons=('djf', 'mam', 'jja', 'son')):
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
        dt = coord.units.num2date(value)
        return month_season_numbers[dt.month]

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season_number)


def add_season_year(cube, coord, name='season_year',
                    seasons=('djf', 'mam', 'jja', 'son')):
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
        dt = coord.units.num2date(value)
        year = dt.year
        year += month_year_adjusts[dt.month]
        return year

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season_year)


def add_season_membership(cube, coord, season, name='season_membership'):
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
        dt = coord.units.num2date(value)
        if dt.month in months:
            return True
        return False

    add_categorised_coord(cube, name, coord, _season_membership)
