# (C) British Crown Copyright 2010 - 2012, Met Office
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

def add_year(cube, coord, name='year'):
    """Add a categorical calendar-year coordinate."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).year
        )


def add_month_number(cube, coord, name='month'):
    """Add a categorical month coordinate, values 1..12."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).month
        )


def add_month_shortname(cube, coord, name='month'):
    """Add a categorical month coordinate, values 'jan'..'dec'."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.month_abbr[_pt_date(coord, x).month],
        units='no_unit'
        )


def add_month_fullname(cube, coord, name='month'):
    """Add a categorical month coordinate, values 'January'..'December'."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.month_name[_pt_date(coord, x).month],
        units='no_unit'
        )


# short alias for default representation
def add_month(cube, coord, name='month'):
    """Alias for :func:`add_month_shortname`."""
    add_month_shortname(cube, coord, name)


def add_day_of_month(cube, coord, name='day'):
    """Add a categorical day-of-month coordinate, values 1..31."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).day
        )


#--------------------------------------------
# time categorisations : days of the week
 
def add_weekday_number(cube, coord, name='weekday'):
    """Add a categorical weekday coordinate, values 0..6  [0=Monday]."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: _pt_date(coord, x).weekday()
        )


def add_weekday_shortname(cube, coord, name='weekday'):
    """Add a categorical weekday coordinate, values 'Mon'..'Sun'."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.day_abbr[_pt_date(coord, x).weekday()],
        units='no_unit'
        )


def add_weekday_fullname(cube, coord, name='weekday'):
    """Add a categorical weekday coordinate, values 'Monday'..'Sunday'."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: calendar.day_name[_pt_date(coord, x).weekday()],
        units='no_unit'
        )


# short alias for default representation
def add_weekday(cube, coord, name='weekday'):
    """Alias for :func:`add_weekday_shortname`."""
    add_weekday_shortname(cube, coord, name)


#--------------------------------------------
# time categorisations : meteorological seasons

#
# some useful data arrays
# - note: those indexed by-month have blank at [0], as with standard-library 'calendar.month_name' et al
#
MONTH_SEASON_NUMBERS = [None, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
"""
Season numbers (0..3) for each month (1..12).

Note:  December is part of the 'next' year
"""


_MONTH_YEAR_ADJUSTS = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   #December is part of the 'next' year


SEASON_MONTHS_INITIALS = ['djf', 'mam', 'jja', 'son']
"""Season month strings (3-character strings) for each season (0..3)."""

 
def add_season_number(cube, coord, name='season'):
    """Add a categorical season-of-year coordinate, values 0..3  [0=djf, 1=mam, ...]."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: MONTH_SEASON_NUMBERS[ _pt_date(coord, x).month ]
        )

  
def add_season_month_initials(cube, coord, name='season'):
    """Add a categorical season-of-year coordinate, values 'djf'..'son'."""
    add_categorised_coord(
        cube, name, coord,
        lambda coord, x: SEASON_MONTHS_INITIALS[ MONTH_SEASON_NUMBERS[_pt_date(coord, x).month] ],
        units='no_unit'
        )

    
# short alias for default representation
def add_season(cube, coord, name='season'):
    """Alias for :func:`add_season_month_initials`."""
    add_season_month_initials(cube, coord, name=name)

    
def add_season_year(cube, coord, name='year'):
    """
    Add a categorical year-of-season coordinate (e.g. Aug'01 -> 1, but Dec'01 -> 2).
    
    Differs from calendar year, because December belongs to a season in the *following* year.'
    """
    def _season_year(coord, value):
        value_date = _pt_date(coord, value)
        year = value_date.year
        year += _MONTH_YEAR_ADJUSTS[value_date.month]
        return year
      
    add_categorised_coord(
        cube, name, coord,
        _season_year
        )
