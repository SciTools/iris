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
import warnings

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
from iris._deprecation import warn_deprecated
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import Cube, CubeList


def _get_dimensional_metadata(name, values, calendar=None, dm_class=None):
    """
    Create a Coord or other dimensional metadata from a Pandas index or columns array.

    If no calendar is specified for a time series, Standard is assumed.

    """
    units = Unit("unknown")
    if calendar is None:
        calendar = cf_units.CALENDAR_STANDARD

    # Getting everything into a single datetime format is hard!

    # Convert out of NumPy's own datetime format.
    if np.issubdtype(values.dtype, np.datetime64):
        values = pandas.to_datetime(values)

    # Convert pandas datetime objects to python datetime objects.
    if isinstance(values, DatetimeIndex):
        values = np.array([i.to_pydatetime() for i in values])

    # Convert datetime objects to Iris' current datetime representation.
    if values.dtype == object:
        dt_types = (datetime.datetime, cftime.datetime)
        if all([isinstance(i, dt_types) for i in values]):
            units = Unit("hours since epoch", calendar=calendar)
            values = units.date2num(values)

    values = np.array(values)

    if dm_class is None:
        if np.issubdtype(values.dtype, np.number) and iris.util.monotonic(
            values, strict=True
        ):
            dm_class = DimCoord
        else:
            dm_class = AuxCoord

    instance = dm_class(values, units=units)
    if name is not None:
        # Use rename() to attempt standard_name but fall back on long_name.
        instance.rename(str(name))

    return instance


def _add_iris_coord(cube, name, points, dim, calendar=None):
    """
    Add a Coord or other dimensional metadata to a Cube from a Pandas index or columns array.
    """
    # Most functionality has been abstracted to _get_dimensional_metadata,
    #  allowing re-use in as_cube() and as_cubes().
    coord = _get_dimensional_metadata(name, points, calendar)

    if coord.__class__ == DimCoord:
        cube.add_dim_coord(coord, dim)
    else:
        cube.add_aux_coord(coord, dim)


def _series_index_unique(pandas_series: pandas.Series):
    """
    Find an index grouping of a :class:`pandas.Series` that has just one Series value per group.

    Iterates through grouping single index levels, then combinations of 2
    levels, then 3 etcetera, until single :class:`~pandas.Series` values per
    group are found. Returns a ``tuple`` of the index levels that group to
    produce single values, as soon as one is found.

    Returns ``None`` if no index level combination produces single values.

    """
    unique_number = pandas_series.nunique()
    pandas_index = pandas_series.index
    levels_range = range(pandas_index.nlevels)
    if unique_number == 1:
        # Scalar - identical for all indices.
        result = ()
    else:
        result = None
        levels_combinations = chain(
            *[
                combinations(levels_range, levels + 1)
                for levels in levels_range
            ]
        )
        for lc in levels_combinations:
            if pandas_series.groupby(level=lc).nunique().max() == 1:
                result = lc
                # Escape as early as possible - heavy operation.
                break
    return result


def as_cube(
    pandas_array,
    copy=True,
    calendars=None,
):
    """
    Convert a Pandas Series/DataFrame into a 1D/2D Iris Cube.

    .. deprecated:: 3.3.0

        This function is scheduled for removal in a future release, being
        replaced by :func:`iris.pandas.as_cubes`, which offers richer
        dimensional intelligence.

    Parameters
    ----------
    pandas_array : :class:`pandas.Series` or :class:`pandas.DataFrame`
        The Pandas object to convert
    copy : bool, default=True
        Whether to copy `pandas_array`, or to create array views where
        possible. Provided in case of memory limit concerns.
    calendars : dict, optional
        A dict mapping a dimension to a calendar. Required to convert datetime
        indices/columns.

    Notes
    -----
    This function will copy your data by default.

    Example usage::

        as_cube(series, calendars={0: cf_units.CALENDAR_360_DAY})
        as_cube(data_frame, calendars={1: cf_units.CALENDAR_STANDARD})

    """
    message = (
        "iris.pandas.as_cube has been deprecated, and will be removed in a "
        "future release. Please use iris.pandas.as_cubes instead."
    )
    warn_deprecated(message)

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


def as_cubes(
    pandas_structure,
    copy=True,
    calendars=None,
    aux_coord_cols=None,
    cell_measure_cols=None,
    ancillary_variable_cols=None,
):
    """
    Convert a Pandas Series/DataFrame into n-dimensional Iris Cubes, including dimensional metadata.

    The index of `pandas_structure` will be used for generating the
    :class:`~iris.cube.Cube` dimension(s) and :class:`~iris.coords.DimCoord`\\ s.
    Other dimensional metadata may span multiple dimensions - based on how the
    column values vary with the index values.

    Parameters
    ----------
    pandas_structure : :class:`pandas.Series` or :class:`pandas.DataFrame`
        The Pandas object to convert
    copy : bool, default=True
        Whether the Cube :attr:`~iris.cube.Cube.data` is a copy of the
        `pandas_structure` column, or a view of the same array. Arrays other than
        the data (coords etc.) are always copies. This option is provided to
        help with memory size concerns.
    calendars : dict, optional
        Calendar conversions for individual date-time coordinate
        columns/index-levels e.g. ``{"my_column": cf_units.CALENDAR_360_DAY}``.
    aux_coord_cols, cell_measure_cols, ancillary_variable_cols : list of str, optional
        Names of columns to be converted into :class:`~iris.coords.AuxCoord`,
        :class:`~iris.coords.CellMeasure` and
        :class:`~iris.coords.AncillaryVariable` objects.

    Returns
    --------
    :class:`~iris.cube.CubeList`
        One :class:`~iris.cube.Cube` for each column not referenced in
        `aux_coord_cols`/`cell_measure_cols`/`ancillary_variable_cols`.

    Notes
    -----
    A :class:`~pandas.DataFrame` using columns as a second data dimension will
    need to be 'melted' before conversion. See the Examples for how.

    Dask ``DataFrame``\\s are not supported.

    Examples
    --------
    >>> from iris.pandas import as_cubes
    >>> import numpy as np
    >>> from pandas import DataFrame, Series

    Converting a simple :class:`~pandas.Series` :

    >>> my_series = Series([300, 301, 302], name="air_temperature")
    >>> converted_cubes = as_cubes(my_series)
    >>> print(converted_cubes)
    0: air_temperature / (unknown)         (unknown: 3)
    >>> print(converted_cubes[0])
    air_temperature / (unknown)         (unknown: 3)
        Dimension coordinates:
            unknown                             x

    A :class:`~pandas.DataFrame`, with a custom index becoming the
    :class:`~iris.coords.DimCoord` :

    >>> my_df = DataFrame({
    ...     "air_temperature": [300, 301, 302],
    ...     "longitude": [30, 40, 50]
    ...     })
    >>> my_df = my_df.set_index("longitude")
    >>> converted_cubes = as_cubes(my_df)
    >>> print(converted_cubes[0])
    air_temperature / (unknown)         (longitude: 3)
        Dimension coordinates:
            longitude                             x

    A :class:`~pandas.DataFrame` representing two 3-dimensional datasets,
    including a 2-dimensional :class:`~iris.coords.AuxCoord` :

    >>> my_df = DataFrame({
    ...     "air_temperature": np.arange(300, 312, 1),
    ...     "air_pressure": np.arange(1000, 1012, 1),
    ...     "longitude": [0, 10] * 6,
    ...     "latitude": [25, 25, 35, 35] * 3,
    ...     "height": ([0] * 4) + ([100] * 4) + ([200] * 4),
    ...     "in_region": [True, False, False, False] * 3
    ... })
    >>> print(my_df)
        air_temperature  air_pressure  longitude  latitude  height  in_region
    0               300          1000          0        25       0       True
    1               301          1001         10        25       0      False
    2               302          1002          0        35       0      False
    3               303          1003         10        35       0      False
    4               304          1004          0        25     100       True
    5               305          1005         10        25     100      False
    6               306          1006          0        35     100      False
    7               307          1007         10        35     100      False
    8               308          1008          0        25     200       True
    9               309          1009         10        25     200      False
    10              310          1010          0        35     200      False
    11              311          1011         10        35     200      False
    >>> my_df = my_df.set_index(["longitude", "latitude", "height"])
    >>> my_df = my_df.sort_index()
    >>> converted_cubes = as_cubes(my_df, aux_coord_cols=["in_region"])
    >>> print(converted_cubes)
    0: air_temperature / (unknown)         (longitude: 2; latitude: 2; height: 3)
    1: air_pressure / (unknown)            (longitude: 2; latitude: 2; height: 3)
    >>> print(converted_cubes[0])
    air_temperature / (unknown)         (longitude: 2; latitude: 2; height: 3)
        Dimension coordinates:
            longitude                             x            -          -
            latitude                              -            x          -
            height                                -            -          x
        Auxiliary coordinates:
            in_region                             x            x          -

    Pandas uses ``NaN`` rather than masking data. Converted
    :class:`~iris.cube.Cube`\\s can be masked in downstream user code :

    >>> my_series = Series([300, np.NaN, 302], name="air_temperature")
    >>> converted_cube = as_cubes(my_series)[0]
    >>> print(converted_cube.data)
    [300.  nan 302.]
    >>> converted_cube.data = np.ma.masked_invalid(converted_cube.data)
    >>> print(converted_cube.data)
    [300.0 -- 302.0]

    If the :class:`~pandas.DataFrame` uses columns as a second dimension,
    :func:`pandas.melt` should be used to convert the data to the expected
    n-dimensional format :

    >>> my_df = DataFrame({
    ...     "latitude": [35, 25],
    ...     0: [300, 301],
    ...     10: [302, 303],
    ... })
    >>> print(my_df)
       latitude    0   10
    0        35  300  302
    1        25  301  303
    >>> my_df = my_df.melt(
    ...     id_vars=["latitude"],
    ...     value_vars=[0, 10],
    ...     var_name="longitude",
    ...     value_name="air_temperature"
    ... )
    >>> print(my_df)
       latitude longitude  air_temperature
    0        35         0              300
    1        25         0              301
    2        35        10              302
    3        25        10              303
    >>> my_df = my_df.set_index(["latitude", "longitude"])
    >>> my_df = my_df.sort_index()
    >>> converted_cube = as_cubes(my_df)[0]
    >>> print(converted_cube)
    air_temperature / (unknown)         (latitude: 2; longitude: 2)
        Dimension coordinates:
            latitude                             x             -
            longitude                            -             x

    """
    if pandas_structure.empty:
        return CubeList()

    calendars = calendars or {}
    aux_coord_cols = aux_coord_cols or []
    cell_measure_cols = cell_measure_cols or []
    ancillary_variable_cols = ancillary_variable_cols or []

    is_series = isinstance(pandas_structure, pandas.Series)

    if copy:
        pandas_structure = pandas_structure.copy()

    pandas_index = pandas_structure.index
    if not pandas_index.is_unique:
        message = (
            f"DataFrame index ({pandas_index.names}) is not unique per "
            "row; cannot be used for DimCoords."
        )
        raise ValueError(message)

    if not pandas_index.is_monotonic:
        # Need monotonic index for use in DimCoord(s).
        # This function doesn't sort_index itself since that breaks the
        #  option to return a data view instead of a copy.
        message = (
            "Pandas index is not monotonic. Consider using the "
            "sort_index() method before passing in."
        )
        raise ValueError(message)

    cube_shape = getattr(pandas_index, "levshape", (pandas_index.nunique(),))
    n_rows = len(pandas_structure)
    if np.product(cube_shape) > n_rows:
        message = (
            f"Not all index values have a corresponding row - {n_rows} rows "
            f"cannot be reshaped into {cube_shape}. Consider padding with NaN "
            "rows where needed."
        )
        raise ValueError(message)

    cube_kwargs = {}

    def format_dimensional_metadata(dm_class_, values_, name_, dimensions_):
        # Common convenience to get the right DM in the right format for
        #  Cube creation.
        calendar = calendars.get(name_)
        instance = _get_dimensional_metadata(
            name_, values_, calendar, dm_class_
        )
        return (instance, dimensions_)

    # DimCoords.
    dim_coord_kwarg = []
    for ix, dim_name in enumerate(pandas_index.names):
        if hasattr(pandas_index, "levels"):
            coord_points = pandas_index.levels[ix]
        else:
            coord_points = pandas_index
        new_dim_coord = format_dimensional_metadata(
            DimCoord, coord_points, dim_name, ix
        )
        dim_coord_kwarg.append(new_dim_coord)
    cube_kwargs["dim_coords_and_dims"] = dim_coord_kwarg

    # Other dimensional metadata.
    class_arg_mapping = [
        (AuxCoord, aux_coord_cols, "aux_coords_and_dims"),
        (CellMeasure, cell_measure_cols, "cell_measures_and_dims"),
        (
            AncillaryVariable,
            ancillary_variable_cols,
            "ancillary_variables_and_dims",
        ),
    ]

    if is_series:
        columns_ignored = any([len(t[1]) > 0 for t in class_arg_mapping])
        if columns_ignored:
            ignored_args = ", ".join([t[2] for t in class_arg_mapping])
            message = f"The input pandas_structure is a Series; ignoring arguments: {ignored_args} ."
            warnings.warn(message)
        class_arg_mapping = []

    non_data_names = []
    for dm_class, column_names, kwarg in class_arg_mapping:
        class_kwarg = []
        non_data_names.extend(column_names)
        for column_name in column_names:
            column = pandas_structure[column_name]

            # Should be impossible for None to be returned - would require a
            #  non-unique index, which we protect against.
            dimensions = _series_index_unique(column)

            content = column.to_numpy()
            # Remove duplicate entries to get down to the correct dimensions
            #  for this object. _series_index_unique should have ensured
            #  that we are indeed removing the duplicates.
            shaped = content.reshape(cube_shape)
            indices = [0] * len(cube_shape)
            for dim in dimensions:
                indices[dim] = slice(None)
            collapsed = shaped[tuple(indices)]

            new_dm = format_dimensional_metadata(
                dm_class, collapsed, column_name, dimensions
            )
            class_kwarg.append(new_dm)

        cube_kwargs[kwarg] = class_kwarg

    # Cube creation.
    if is_series:
        data_series_list = [pandas_structure]
    else:
        data_series_list = [
            pandas_structure[column_name]
            for column_name in pandas_structure.columns
            if column_name not in non_data_names
        ]
    cubes = CubeList()
    for data_series in data_series_list:
        cube_data = data_series.to_numpy().reshape(cube_shape)
        new_cube = Cube(cube_data, **cube_kwargs)
        if data_series.name is not None:
            # Use rename() to attempt standard_name but fall back on long_name.
            new_cube.rename(str(data_series.name))
        cubes.append(new_cube)

    return cubes


def _as_pandas_coord(coord):
    """Convert an Iris Coord into a Pandas index or columns array."""
    index = coord.points
    if coord.units.is_time_reference():
        index = coord.units.num2date(index)
    return index


def _assert_shared(np_obj, pandas_obj):
    """Ensure the pandas object shares memory."""
    values = pandas_obj.values

    def _get_base(array):
        # Chase the stack of NumPy `base` references back to the original array
        while array.base is not None:
            array = array.base
        return array

    base = _get_base(values)
    np_base = _get_base(np_obj)
    if base is not np_base:
        msg = "Pandas {} does not share memory".format(
            type(pandas_obj).__name__
        )
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
    if ma.isMaskedArray(data):
        if not copy:
            raise ValueError("Masked arrays must always be copied.")
        data = data.astype("f").filled(np.nan)
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
    if ma.isMaskedArray(data):
        if not copy:
            raise ValueError("Masked arrays must always be copied.")
        data = data.astype("f").filled(np.nan)
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
