# (C) British Crown Copyright 2013 - 2014, Met Office
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
Provide conversion to and from Pandas data structures.

See also: http://pandas.pydata.org/

"""
from __future__ import absolute_import

import datetime

import netcdftime
import numpy as np
import pandas

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.unit import Unit


def _add_iris_coord(cube, name, points, dim, calendar=None):
    """
    Add a Coord to a Cube from a Pandas index or columns array.

    If no calendar is specified for a time series, Gregorian is assumed.

    """
    units = Unit("unknown")
    if calendar is None:
        calendar = iris.unit.CALENDAR_GREGORIAN

    # Convert pandas datetime objects to python datetime obejcts.
    if isinstance(points, pandas.tseries.index.DatetimeIndex):
        points = np.array([i.to_datetime() for i in points])

    # Convert datetime objects to Iris' current datetime representation.
    if points.dtype == object:
        dt_types = (datetime.datetime, netcdftime.datetime)
        if all([isinstance(i, dt_types) for i in points]):
            units = Unit("hours since epoch", calendar=calendar)
            points = units.date2num(points)

    points = np.array(points)
    if (np.issubdtype(points.dtype, np.number) and
            iris.util.monotonic(points, strict=True)):
                coord = DimCoord(points, units=units)
                coord.rename(name)
                cube.add_dim_coord(coord, dim)
    else:
        coord = AuxCoord(points, units=units)
        coord.rename(name)
        cube.add_aux_coord(coord, dim)


def as_cube(pandas_array, copy=True, calendars=None):
    """
    Convert a Pandas array into an Iris cube.

    Args:

        * pandas_array - A Pandas Series or DataFrame.

    Kwargs:

        * copy      - Whether to make a copy of the data.
                      Defaults to True.

        * calendars - A dict mapping a dimension to a calendar.
                      Required to convert datetime indices/columns.

    Example usage::

        as_cube(series, calendars={0: iris.unit.CALENDAR_360_DAY})
        as_cube(data_frame, calendars={1: iris.unit.CALENDAR_GREGORIAN})

    .. note:: This function will copy your data by default.

    """
    calendars = calendars or {}
    if pandas_array.ndim not in [1, 2]:
        raise ValueError("Only 1D or 2D Pandas arrays "
                         "can currently be conveted to Iris cubes.")

    cube = Cube(np.ma.masked_invalid(pandas_array, copy=copy))
    _add_iris_coord(cube, "index", pandas_array.index, 0,
                    calendars.get(0, None))
    if pandas_array.ndim == 2:
        _add_iris_coord(cube, "columns", pandas_array.columns, 1,
                        calendars.get(1, None))
    return cube


def _as_pandas_coord(coord):
    """Convert an Iris Coord into a Pandas index or columns array."""
    index = coord.points
    if coord.units.is_time_reference():
        index = coord.units.num2date(index)
    return index


def _assert_shared(np_obj, pandas_obj):
    """Ensure the pandas object shares memory."""
    if hasattr(pandas_obj, 'base'):
        base = pandas_obj.base
    else:
        base = pandas_obj[0].base
    # Chase the stack of NumPy `base` references back to see if any of
    # them are our original array.
    while base is not None:
        if base is np_obj:
            return
        # Take the next step up the stack of `base` references.
        base = base.base
    msg = 'Pandas {} does not share memory'.format(type(pandas_obj).__name__)
    raise AssertionError(msg)


def as_series(cube, copy=True):
    """
    Convert a 1D cube to a Pandas Series.

    Args:

        * cube - The cube to convert to a Pandas Series.

    Kwargs:

        * copy - Whether to make a copy of the data.
                 Defaults to True. Must be True for masked data.

    .. note::

        This function will copy your data by default.
        If you have a large array that cannot be copied,
        make sure it is not masked and use copy=False.

    """
    data = cube.data
    if isinstance(data, np.ma.MaskedArray):
        if not copy:
            raise ValueError("Masked arrays must always be copied.")
        data = data.astype('f').filled(np.nan)
    elif copy:
        data = data.copy()

    index = None
    if cube.dim_coords:
        index = _as_pandas_coord(cube.dim_coords[0])

    series = pandas.Series(data, index)
    if not copy:
        _assert_shared(data, series)

    return series


def as_data_frame(cube, copy=True):
    """
    Convert a 2D cube to a Pandas DataFrame.

    Args:

        * cube - The cube to convert to a Pandas DataFrame.

    Kwargs:

        * copy - Whether to make a copy of the data.
                 Defaults to True. Must be True for masked data
                 and some data types (see notes below).

    .. note::

        This function will copy your data by default.
        If you have a large array that cannot be copied,
        make sure it is not masked and use copy=False.

    .. note::

        Pandas will sometimes make a copy of the array,
        for example when creating from an int32 array.
        Iris will detect this and raise an exception if copy=False.

    """
    data = cube.data
    if isinstance(data, np.ma.MaskedArray):
        if not copy:
            raise ValueError("Masked arrays must always be copied.")
        data = data.astype('f').filled(np.nan)
    elif copy:
        data = data.copy()

    index = columns = None
    if cube.coords(dimensions=[0]):
        index = _as_pandas_coord(cube.coord(dimensions=[0]))
    if cube.coords(dimensions=[1]):
        columns = _as_pandas_coord(cube.coord(dimensions=[1]))

    data_frame = pandas.DataFrame(data, index, columns)
    if not copy:
        _assert_shared(data, data_frame)

    return data_frame
