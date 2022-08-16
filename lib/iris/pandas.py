# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provide conversion to and from Pandas data structures.

See also: http://pandas.pydata.org/

"""

import datetime

import cf_units
from cf_units import Unit
import cftime
import numpy as np
import numpy.ma as ma
import pandas

try:
    from pandas.core.indexes.datetimes import DatetimeIndex  # pandas >=0.20
except ImportError:
    from pandas.tseries.index import DatetimeIndex  # pandas <0.20

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


def _add_iris_coord(cube, name, points, dim, calendar=None):
    """
    Add a Coord to a Cube from a Pandas index or columns array.

    If no calendar is specified for a time series, Standard is assumed.

    """
    units = Unit("unknown")
    if calendar is None:
        calendar = cf_units.CALENDAR_STANDARD

    # Convert pandas datetime objects to python datetime obejcts.
    if isinstance(points, DatetimeIndex):
        points = np.array([i.to_pydatetime() for i in points])

    # Convert datetime objects to Iris' current datetime representation.
    if points.dtype == object:
        dt_types = (datetime.datetime, cftime.datetime)
        if all([isinstance(i, dt_types) for i in points]):
            units = Unit("hours since epoch", calendar=calendar)
            points = units.date2num(points)

    points = np.array(points)
    if np.issubdtype(points.dtype, np.number) and iris.util.monotonic(
        points, strict=True
    ):
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

        as_cube(series, calendars={0: cf_units.CALENDAR_360_DAY})
        as_cube(data_frame, calendars={1: cf_units.CALENDAR_STANDARD})

    .. note:: This function will copy your data by default.

    """
    calendars = calendars or {}
    if pandas_array.ndim not in [1, 2]:
        raise ValueError(
            "Only 1D or 2D Pandas arrays "
            "can currently be conveted to Iris cubes."
        )

    # Make the copy work consistently across NumPy 1.6 and 1.7.
    # (When 1.7 takes a copy it preserves the C/Fortran ordering, but
    # 1.6 doesn't. Since we don't care about preserving the order we can
    # just force it back to C-order.)
    order = "C" if copy else "A"
    data = np.array(pandas_array, copy=copy, order=order)
    cube = Cube(np.ma.masked_invalid(data, copy=False))
    _add_iris_coord(
        cube, "index", pandas_array.index, 0, calendars.get(0, None)
    )
    if pandas_array.ndim == 2:
        _add_iris_coord(
            cube,
            "columns",
            pandas_array.columns.values,
            1,
            calendars.get(1, None),
        )
    return cube


def _as_pandas_coord(coord):
    """Convert an Iris Coord into a Pandas index or columns array."""
    index = coord.points
    if coord.units.is_time_reference():
        index = coord.units.num2date(index)
    return index


def as_data_frame(
    cube,
    copy=False,
    asmultiindex=False,
    add_aux_coord=False,
    add_global_attributes=None,
):
    """
    Convert a :class:`Cube` to a Pandas `DataFrame`.

    Args:

        * cube:
            The :class:`Cube` to be converted to a Pandas `DataFrame`.

    Kwargs:

        * copy - Whether to make a copy of the data. Defaults to False.
        * asmultiindex:
            If True, returns a `DataFrame` with a `MultiIndex <https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.html#pandas.MultiIndex>`_.  Defaults to False.
        * add_aux_coord:
            If True, add all :class:`Cube` :class:`AuxCoord`s to add to the returned `DataFrame`. Defaults to False.
        * add_global_attributes:
            A list of :class:`Cube` attributes to add to the returned `DataFrame`. Defaults to None.

    .. note::

        By default, this function seeks to minimise data copying such that the new `DataFrame` shares a reference the `cube.data` values. If you want to break the reference between :class:`Cube` and `DataFrame` select `copy=True`.

    """
    if copy:
        data = cube.data.copy()
    else:
        data = cube.data
    if ma.isMaskedArray(data):
        data = data.astype("f").filled(np.nan)

    # Extract dim coord information
    if cube.ndim != len(cube.dim_coords):
        # Create dummy dim coord information if dim coords not defined
        coord_names = ["dim" + str(n) for n in range(cube.ndim)]
        coords = [range(dim) for dim in cube.shape]
        for onecoord in cube.dim_coords:
            for i, dummyc in enumerate(coords):
                if len(dummyc) == len(onecoord.points):
                    coords[i] = _as_pandas_coord(onecoord)
                    coord_names[i] = onecoord.name()
                else:
                    pass
    else:
        coord_names = [x.name() for x in cube.dim_coords]
        coords = [_as_pandas_coord(x) for x in cube.dim_coords]

    index = pandas.MultiIndex.from_product(coords, names=coord_names)
    data_frame = pandas.DataFrame(
        data.ravel(), columns=[cube.name()], index=index
    )

    # Add aux coord information
    if add_aux_coord:
        aux_coord_names = [x.name() for x in cube.aux_coords]
        for acoord in aux_coord_names:
            aux_coord = cube.coord(acoord)
            which_dim = cube.shape.index(aux_coord.shape[0])
            aux_coord_index = coords[
                which_dim
            ]  # Make corresponding dim coord as aux coord index
            # Build aux coord dataframe
            acoord_df = pandas.DataFrame(
                {acoord: aux_coord.points},
                index=pandas.Index(
                    data=aux_coord_index,
                    name=coord_names[which_dim],
                ),
            )
            # Merge to main data frame
            data_frame = pandas.merge(
                data_frame, acoord_df, on=coord_names[which_dim]
            )

    # Add data from global attributes
    if add_global_attributes:
        global_attribute_names = list(cube.attributes.keys())
        for global_attribute in add_global_attributes:
            if (
                global_attribute not in global_attribute_names
            ):  # Check global attribute exists
                raise ValueError(f'"{global_attribute}" not in cube')
            else:
                data_frame[global_attribute] = cube.attributes[
                    global_attribute
                ]

    if not asmultiindex:
        data_frame.reset_index(inplace=True)

    return data_frame
