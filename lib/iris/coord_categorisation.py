# (C) British Crown Copyright 2010 - 2013, Met Office
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
    * The function :func:`add_categorised_coord` performs a generic coordinate categorisation.
    * The other functions all implement specific common cases (e.g. :func:`add_day_of_month`).
      Currently, these are all calendar functions, so they only apply to "Time coordinates".

"""

import calendar   #for day and month names
import collections   # for counting months when validating seasons
import warnings   # temporary for deprecations
import functools   # temporary for deprecations

import iris.coords


def add_categorised_coord(cube, name, from_coord, category_function, units='1'):
    """
    Add a new coordinate to a cube, by categorising an existing one.

    Make a new :class:`iris.coords.AuxCoord` from mapped values, and add it to the cube.

    Args:

    * cube (:class:`iris.cube.Cube`):
        the cube containing 'from_coord'.  The new coord will be added into it.
    * name (string):
        name of the created coordinate
    * from_coord (:class:`iris.coords.Coord` or string):
        coordinate in 'cube', or the name of one
    * category_function (callable):
        function(coordinate, value), returning a category value for a coordinate point-value

    Kwargs:

    * units:
        units of the category value, typically 'no_unit' or '1'.
    """
    #interpret coord, if given as a name
    if isinstance(from_coord, basestring):
        from_coord = cube.coord(from_coord)

    if len(cube.coords(name)) > 0:
        raise ValueError('A coordinate "%s" already exists in the cube.' % name)

    #construct new coordinate by mapping values
    points = [category_function(from_coord, value) for value in from_coord.points]
    new_coord = iris.coords.AuxCoord(points, units=units, attributes=from_coord.attributes.copy())
    new_coord.rename(name)

    # add into the cube
    cube.add_aux_coord(new_coord, cube.coord_dims(from_coord))


#======================================
# Specific functions for particular purposes
#
# NOTE: all the existing ones are calendar operations, so are for 'Time' coordinates only
#

# private "helper" function
def _pt_date(coord, time):
    """
    Return the date of a time-coordinate point.

    Args:

    * coord (Coord):
        coordinate (must be Time-type)
    * time (float):
        value of a coordinate point

    Returns:
        datetime.date
    """
    # NOTE: all of the currently defined categorisation functions are calendar operations on Time coordinates
    #  - all these currently depend on Unit::num2date, which is deprecated (!!)
    #  - we will want to do better, when we sort out our own Calendars
    #  - for now, just make sure these all call through this one function
    return coord.units.num2date(time)


#--------------------------------------------
# time categorisations : calendar date components

# This is a temporary helper function to manage the transition away from
# ambiguous default values. It was first released in 1.4.
def _check_default(name, deprecated_default, upcoming_default):
    if name is None:
        msg = 'Default value for `name` will change from {0!r} to {1!r}'
        msg = msg.format(deprecated_default, upcoming_default)
        warnings.warn(msg, stacklevel=3)
        name = deprecated_default
    return name


def add_year(cube, coord, name='year'):
    """Add a categorical calendar-year coordinate."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).year
        )


def add_month_number(cube, coord, name=None):
    """Add a categorical month coordinate, values 1..12."""
    name = _check_default(name, 'month', 'month_number')
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).month
        )


def add_month_shortname(cube, coord, name='month'):
    """
    Add a categorical month coordinate, values 'Jan'..'Dec'.

        .. deprecated:: 1.4
            Please use :func:`~iris.coord_categorisation.add_month()`.

    """
    msg = "The 'add_month_shortname()' function is deprecated." \
          " Please use 'add_month()' instead."
    warnings.warn(msg, UserWarning, stacklevel=2)
    add_month(cube, coord, name)


def add_month_fullname(cube, coord, name=None):
    """Add a categorical month coordinate, values 'January'..'December'."""
    name = _check_default(name, 'month', 'month_fullname')
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.month_name[_pt_date(coord, x).month],
        units='no_unit'
        )


def add_month(cube, coord, name='month'):
    """Add a categorical month coordinate, values 'Jan'..'Dec'."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.month_abbr[_pt_date(coord, x).month],
        units='no_unit'
        )


def add_day_of_month(cube, coord, name=None):
    """Add a categorical day-of-month coordinate, values 1..31."""
    name = _check_default(name, 'day', 'day_of_month')
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).day
        )


def add_day_of_year(cube, coord, name=None):
    """
    Add a categorical day-of-year coordinate, values 1..365
    (1..366 in leap years).

    """
    name = _check_default(name, 'day', 'day_of_year')
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).timetuple().tm_yday)


#--------------------------------------------
# time categorisations : days of the week

def add_weekday_number(cube, coord, name=None):
    """Add a categorical weekday coordinate, values 0..6  [0=Monday]."""
    name = _check_default(name, 'weekday', 'weekday_number')
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).weekday()
        )


def add_weekday_shortname(cube, coord, name='weekday'):
    """
    Add a categorical weekday coordinate, values 'Mon'..'Sun'.

        .. deprecated:: 1.4
            Please use :func:`~iris.coord_categorisation.add_weekday()`.

    """
    msg = "The 'add_weekday_shortname()' function is deprecated." \
          " Please use 'add_weekday()' instead."
    warnings.warn(msg, UserWarning, stacklevel=2)
    add_weekday(cube, coord, name)


def add_weekday_fullname(cube, coord, name=None):
    """Add a categorical weekday coordinate, values 'Monday'..'Sunday'."""
    name = _check_default(name, 'weekday', 'weekday_fullname')
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.day_name[_pt_date(coord, x).weekday()],
        units='no_unit'
        )


def add_weekday(cube, coord, name='weekday'):
    """Add a categorical weekday coordinate, values 'Mon'..'Sun'."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.day_abbr[_pt_date(coord, x).weekday()],
        units='no_unit'
        )


#----------------------------------------------
# time categorisations : meteorological seasons

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
    return map(lambda month: (month % 12) + 1, range(m0, m1))


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
    # make a list of months that are not present...
    not_present = [calendar.month_abbr[month] for month in xrange(1, 13)
                   if month not in c.keys()]
    if not_present:
        raise ValueError('some months do not appear in any season: '
                         '{!s}'.format(', '.join(not_present)))
    # make a list of months that appear multiple times...
    multi_present = [calendar.month_abbr[month] for month in xrange(1, 13)
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
        for month in filter(lambda m: m > months[-1], months):
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



def add_season_month_initials(cube, coord, name='season'):
    """
    Add a categorical season-of-year coordinate, values 'djf'..'son'.

        .. deprecated:: 1.4
            Please use :func:`~iris.coord_categorisation.add_season()`.

    """
    msg = "The 'add_season_month_initials()' function is deprecated." \
          " Please use 'add_season()' instead."
    warnings.warn(msg, UserWarning, stacklevel=2)
    add_season(cube, coord, name)


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
        dt = _pt_date(coord, value)
        return seasons[month_season_numbers[dt.month]]

    # Apply the categorisation.
    add_categorised_coord(cube, name, coord, _season, units='no_unit')


def add_season_number(cube, coord, name=None,
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
        Name of the created coordinate. Currently defaults to "season",
        but this will change in a later version to "season_number".
    * seasons (:class:`list` of strings):
        List of seasons defined by month abbreviations. Each month must
        appear once and only once. Defaults to standard meteorological
        seasons ('djf', 'mam', 'jja', 'son').

    """
    name = _check_default(name, 'season', 'season_number')
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


def add_season_year(cube, coord, name=None,
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
        Name of the created coordinate. Currently defaults to "year",
        but this will change in a later version to "season_year".
    * seasons (:class:`list` of strings):
        List of seasons defined by month abbreviations. Each month must
        appear once and only once. Defaults to standard meteorological
        seasons ('djf', 'mam', 'jja', 'son').

    """
    name = _check_default(name, 'year', 'season_year')
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


def add_season_membership(cube, coord, season, name=None):
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
        Name of the created coordinate. Currently defaults to "season",
        but this will change in a later version to "season_membership".

    """
    name = _check_default(name, 'season', 'season_membership')

    months = _months_in_season(season)

    def _season_membership(coord, value):
        dt = _pt_date(coord, value)
        if dt.month in months:
            return True
        return False

    add_categorised_coord(cube, name, coord, _season_membership)


#-----------------------------------
# deprecated custom season functions
#

# A temporary decorator to manage the deprecation of the add_custom_season*
# functions. Issues a warning prior to calling the function.
def _custom_season_deprecation(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        msg = "The '{0!s}()' function is deprecated." \
              " Please use '{1!s}()' instead."
        msg = msg.format(func.func_name, func.func_name.replace('_custom', ''))
        warnings.warn(msg, stacklevel=2)
        return func(*args, **kwargs)
    return _wrapper


@_custom_season_deprecation
def add_custom_season(cube, coord, seasons, name='season'):
    """
        .. deprecated:: 1.4
            Please use :func:`~iris.coord_categorisation.add_season()`.

    """
    return add_season(cube, coord, name=name, seasons=seasons)


@_custom_season_deprecation
def add_custom_season_number(cube, coord, seasons, name='season'):
    """
        .. deprecated:: 1.4
            Please use
            :func:`~iris.coord_categorisation.add_season_number()`.

    """
    return add_season_number(cube, coord, name=name, seasons=seasons)


@_custom_season_deprecation
def add_custom_season_year(cube, coord, seasons, name='year'):
    """
        .. deprecated:: 1.4
            Please use
            :func:`~iris.coord_categorisation.add_season_year()`.

    """
    return add_season_year(cube, coord, name=name, seasons=seasons)


@_custom_season_deprecation
def add_custom_season_membership(cube, coord, season, name='season'):
    """
        .. deprecated:: 1.4
            Please use
            :func:`~iris.coord_categorisation.add_season_membership()`.

    """
    return add_season_membership(cube, coord, season, name=name)
