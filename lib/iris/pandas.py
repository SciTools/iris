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
from itertools import chain, combinations

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


def _get_dim_combinations(ndim):
    """Get all possible dim coordinate combinations."""
    dimcomb = []
    for l in range(1, ndim + 1):
        dimcomb += list(combinations(range(ndim), l))
    return dimcomb


def as_data_frame(
    cube,
    copy=True,
    asmultiindex=False,
    add_aux_coord=False,
    add_global_attributes=None,
):
    """
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
    >>> import iris
    >>> from iris.pandas import as_data_frame

    Convert a simple :class:`~iris.cube.Cube` :

    >>> path = iris.sample_data_path('ostia_monthly.nc')
    >>> cube = iris.load_cube(path)
    >>> df = ipd.as_data_frame(cube)
    >>> print(df)
                            time  latitude   longitude  surface_temperature
    0       2006-04-16 00:00:00 -4.999992    0.000000           301.659271
    1       2006-04-16 00:00:00 -4.999992    0.833333           301.785004
    2       2006-04-16 00:00:00 -4.999992    1.666667           301.820984
    3       2006-04-16 00:00:00 -4.999992    2.500000           301.865234
    4       2006-04-16 00:00:00 -4.999992    3.333333           301.926819
    ...                     ...       ...         ...                  ...
    419899  2010-09-16 00:00:00  4.444450  355.833313           298.779938
    419900  2010-09-16 00:00:00  4.444450  356.666656           298.913147
    419901  2010-09-16 00:00:00  4.444450  357.500000                  NaN
    419902  2010-09-16 00:00:00  4.444450  358.333313                  NaN
    419903  2010-09-16 00:00:00  4.444450  359.166656           298.995148

    [419904 rows x 4 columns]

    Using `add_aux_coord=True` maps `~iris.coords.AuxCoord` and scalar coordinate information
    to the `~pandas.DataFrame`:

    >>> df = ipd.as_data_frame(cube, add_aux_coord=True)
    >>> print(df)
                            time  latitude   longitude  surface_temperature  forecast_reference_time  forecast_period
    0       2006-04-16 00:00:00 -4.999992    0.000000           301.659271                 318108.0                0
    1       2006-04-16 00:00:00 -4.999992    0.833333           301.785004                 318108.0                0
    2       2006-04-16 00:00:00 -4.999992    1.666667           301.820984                 318108.0                0
    3       2006-04-16 00:00:00 -4.999992    2.500000           301.865234                 318108.0                0
    4       2006-04-16 00:00:00 -4.999992    3.333333           301.926819                 318108.0                0
    ...                     ...       ...         ...                  ...                      ...              ...
    419899  2010-09-16 00:00:00  4.444450  355.833313           298.779938                 356844.0                0
    419900  2010-09-16 00:00:00  4.444450  356.666656           298.913147                 356844.0                0
    419901  2010-09-16 00:00:00  4.444450  357.500000                  NaN                 356844.0                0
    419902  2010-09-16 00:00:00  4.444450  358.333313                  NaN                 356844.0                0
    419903  2010-09-16 00:00:00  4.444450  359.166656           298.995148                 356844.0                0

    [419904 rows x 6 columns]

    To add netCDF global attribution information to the `~pandas.DataFrame`, specifiy the attribute using the `add_global_attributes`
    keyword:

    >>> df = ipd.as_data_frame(cube, add_aux_coord=True, add_global_attributes=['STASH'])
    >>> print(df)
                            time  latitude   longitude  ...  forecast_reference_time  forecast_period       STASH
    0       2006-04-16 00:00:00 -4.999992    0.000000  ...                 318108.0                0  m01s00i024
    1       2006-04-16 00:00:00 -4.999992    0.833333  ...                 318108.0                0  m01s00i024
    2       2006-04-16 00:00:00 -4.999992    1.666667  ...                 318108.0                0  m01s00i024
    3       2006-04-16 00:00:00 -4.999992    2.500000  ...                 318108.0                0  m01s00i024
    4       2006-04-16 00:00:00 -4.999992    3.333333  ...                 318108.0                0  m01s00i024
    ...                     ...       ...         ...  ...                      ...              ...         ...
    419899  2010-09-16 00:00:00  4.444450  355.833313  ...                 356844.0                0  m01s00i024
    419900  2010-09-16 00:00:00  4.444450  356.666656  ...                 356844.0                0  m01s00i024
    419901  2010-09-16 00:00:00  4.444450  357.500000  ...                 356844.0                0  m01s00i024
    419902  2010-09-16 00:00:00  4.444450  358.333313  ...                 356844.0                0  m01s00i024
    419903  2010-09-16 00:00:00  4.444450  359.166656  ...                 356844.0                0  m01s00i024

    [419904 rows x 7 columns]


    """
    if copy:
        data = cube.data.copy()
    else:
        data = cube.data
    if ma.isMaskedArray(data):
        data = data.astype("f").filled(np.nan)

    # Extract dim coord information
    dim_coord_list = list(
        chain.from_iterable(
            [
                [
                    [n, coord]
                    for coord in cube.coords(dimensions=n, dim_coords=True)
                ]
                for n in range(cube.ndim)
            ]
        )
    )
    # Initalise recieving lists for DataFrame dim information
    coords = list(range(cube.ndim))
    coord_names = list(range(cube.ndim))
    for dim_index, dim_coord in dim_coord_list:
        if not dim_coord:
            # Create dummy dim coord information if dim coords not defined
            coord_names[dim_index] = "dim" + str(dim_index)
            coords[dim_index] = range(cube.shape[dim_index])
        else:
            coord_names[dim_index] = dim_coord.name()
            coords[dim_index] = _as_pandas_coord(dim_coord)

    # Make base DataFrame
    index = pandas.MultiIndex.from_product(coords, names=coord_names)
    data_frame = pandas.DataFrame(
        data.ravel(), columns=[cube.name()], index=index
    )

    # Add AuxCoord & scalar coordinate information
    if add_aux_coord:
        # Extract aux coord information
        aux_coord_list = list(
            chain.from_iterable(
                [
                    [
                        [n, coord]
                        for coord in cube.coords(
                            dimensions=n, dim_coords=False
                        )
                    ]
                    for n in _get_dim_combinations(cube.ndim)
                ]
            )
        )
        for aux_coord_index, aux_coord in aux_coord_list:
            acoord_df = pandas.DataFrame(
                aux_coord.points.ravel(),
                columns=[aux_coord.name()],
                index=pandas.MultiIndex.from_product(
                    [coords[i] for i in aux_coord_index],
                    names=[coord_names[i] for i in aux_coord_index],
                ),
            )
            # Merge to main data frame
            data_frame = pandas.merge(
                data_frame, acoord_df, left_index=True, right_index=True
            )

        # Add scalar coordinate information
        scalar_coord_list = cube.coords(dimensions=(), dim_coords=False)
        for scalar_coord in scalar_coord_list:
            data_frame[scalar_coord.name()] = scalar_coord.points.squeeze()

    # Add global attribute information
    if add_global_attributes:
        global_attribute_names = list(cube.attributes.keys())
        for global_attribute in add_global_attributes:
            if (
                global_attribute not in global_attribute_names
            ):  # Check global attribute exists
                raise ValueError(f'"{global_attribute}" attribute not in cube')
            else:
                if isinstance(
                    cube.attributes[global_attribute],
                    iris.fileformats.pp.STASH,
                ):
                    data_frame[global_attribute] = str(
                        cube.attributes[global_attribute]
                    )
                else:
                    data_frame[global_attribute] = cube.attributes[
                        global_attribute
                    ]

    # Final sort by dim order
    if asmultiindex:
        return data_frame.reorder_levels(coord_names).sort_index()
    else:
        return (
            data_frame.reorder_levels(coord_names).sort_index().reset_index()
        )
