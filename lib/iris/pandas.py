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
    copy=True,
    asmultiindex=False,
    add_aux_coord=False,
    add_global_attributes=None,
):
    """
    Convert an n-dimensional :class:`Cube` to a :class:`~pandas.DataFrame`, including dimensional metadata.

    :attr:`~iris.cube.Cube.dim_coords` and :attr:`~iris.cube.Cube.data` are flattened into a long-style
    :class:`~pandas.DataFrame`.  Other :attr:`~iris.cube.Cube.aux_coords` and :attr:`~iris.cube.Cube.attributes`
    may be optionally added as additional :class:`~pandas.DataFrame` columns.

    Parameters
    ----------
    :class:`~iris.cube.Cube`:
        The :class:`Cube` to be converted to a Pandas `DataFrame`.
    copy: bool, default=True
        Whether the Pandas `DataFrame` is a copy of the the Cube :attr:`~iris.cube.Cube.data`.
        This option is provided to help with memory size concerns.
    asmultiindex: bool, default=False
        If True, returns a `DataFrame` with a `MultiIndex <https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.html#pandas.MultiIndex>`_.
    add_aux_coord: bool, default=False
        If True, add all :attr:`~iris.cube.Cube.aux_coords` to add to the returned `DataFrame`.
    add_global_attributes: list of str, optional
        Names of :attr:`~iris.cube.Cube.attributes` to add to the returned `DataFrame`.

    Returns
    -------
    A :class:`~pandas.DataFrame`

    Notes
    -----
    Dask ``DataFrame``\\s are not supported.

    Examples
    --------
    >>> from iris.pandas import as_data_frame
    >>> import numpy as np
    >>> from pandas import DataFrame

    TODO
    
    """
    if copy:
        data = cube.data.copy()
    else:
        data = cube.data
    if ma.isMaskedArray(data):
        data = data.astype("f").filled(np.nan)

    # Extract dim coord information
    dim_coord_list = [
        cube.coords(dimensions=n, dim_coords=True) for n in range(cube.ndim)
    ]
    # Initalise recieving lists for DataFrame dim information
    coords = list(range(cube.ndim))
    coord_names = list(range(cube.ndim))
    for dim_index, dim_coord in enumerate(dim_coord_list):
        if not dim_coord:
            # Create dummy dim coord information if dim coords not defined
            coord_names[dim_index] = "dim" + str(dim_index)
            coords[dim_index] = range(cube.shape[dim_index])
        else:
            coord_names[dim_index] = dim_coord[0].name()
            coords[dim_index] = _as_pandas_coord(dim_coord[0])

    # Make base DataFrame
    index = pandas.MultiIndex.from_product(coords, names=coord_names)
    data_frame = pandas.DataFrame(
        data.ravel(), columns=[cube.name()], index=index
    )

    # Add aux coord information
    if add_aux_coord:
        for aux_coord_index, aux_coord in enumerate(cube.aux_coords):
            # Build aux coord dataframe with corresponding dim coord as aux coord index
            acoord_df = pandas.Series(
                aux_coord.points,
                name=aux_coord.name(),
                index=pandas.Index(
                    data=coords[aux_coord_index],
                    name=coord_names[aux_coord_index],
                ),
            )
            # Merge to main data frame
            data_frame = pandas.merge(
                data_frame, acoord_df, left_index=True, right_index=True
            )

    # Add global attribute information
    if add_global_attributes:
        global_attribute_names = list(cube.attributes.keys())
        for global_attribute in add_global_attributes:
            if (
                global_attribute not in global_attribute_names
            ):  # Check global attribute exists
                raise ValueError(f'"{global_attribute}" attribute not in cube')
            else:
                data_frame[global_attribute] = cube.attributes[
                    global_attribute
                ]

    if not asmultiindex:
        data_frame.reset_index(inplace=True)

    return data_frame
