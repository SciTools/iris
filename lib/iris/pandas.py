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

    Since this function converts to/from a Pandas object, laziness will not be preserved.

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
            "can currently be converted to Iris cubes."
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

    :class:`dask.dataframe.DataFrame`\\ s are not supported.

    Since this function converts to/from a Pandas object, laziness will not be preserved.

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
    >>> my_df["longitude"] = my_df["longitude"].infer_objects()
    >>> print(my_df)
       latitude  longitude  air_temperature
    0        35          0              300
    1        25          0              301
    2        35         10              302
    3        25         10              303
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

    if not (
        pandas_index.is_monotonic_increasing
        or pandas_index.is_monotonic_decreasing
    ):
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


def _make_dim_coord_list(cube):
    """Get Dimension coordinates."""
    outlist = []
    for dimn in range(cube.ndim):
        dimn_coord = cube.coords(dimensions=dimn, dim_coords=True)
        if dimn_coord:
            outlist += [
                [dimn_coord[0].name(), _as_pandas_coord(dimn_coord[0])]
            ]
        else:
            outlist += [[f"dim{dimn}", range(cube.shape[dimn])]]
    return list(zip(*outlist))


def _make_aux_coord_list(cube):
    """Get Auxiliary coordinates."""
    outlist = []
    for aux_coord in cube.coords(dim_coords=False):
        outlist += [
            [
                aux_coord.name(),
                cube.coord_dims(aux_coord),
                _as_pandas_coord(aux_coord),
            ]
        ]
    return list(chain.from_iterable([outlist]))


def _make_ancillary_variables_list(cube):
    """Get Ancillary variables."""
    outlist = []
    for ancil_var in cube.ancillary_variables():
        outlist += [
            [
                ancil_var.name(),
                cube.ancillary_variable_dims(ancil_var),
                ancil_var.data,
            ]
        ]
    return list(chain.from_iterable([outlist]))


def _make_cell_measures_list(cube):
    """Get cell measures."""
    outlist = []
    for cell_measure in cube.cell_measures():
        outlist += [
            [
                cell_measure.name(),
                cube.cell_measure_dims(cell_measure),
                cell_measure.data,
            ]
        ]
    return list(chain.from_iterable([outlist]))


def as_series(cube, copy=True):
    """
    Convert a 1D cube to a Pandas Series.

    .. deprecated:: 3.4.0
        This function is scheduled for removal in a future release, being
        replaced by :func:`iris.pandas.as_data_frame`, which offers improved
        multi dimension handling.

    Parameters
    ----------
    cube: :class:`Cube`
        The cube to convert to a Pandas Series.
    copy : bool, default=True
        Whether to make a copy of the data.
        Defaults to True. Must be True for masked data.

    Notes
    -----
    This function will copy your data by default.
    If you have a large array that cannot be copied,
    make sure it is not masked and use copy=False.

    Notes
    ------
    Since this function converts to/from a Pandas object, laziness will not be preserved.

    """
    message = (
        "iris.pandas.as_series has been deprecated, and will be removed in a "
        "future release. Please use iris.pandas.as_data_frame instead."
    )
    warn_deprecated(message)

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


def as_data_frame(
    cube,
    copy=True,
    add_aux_coords=False,
    add_cell_measures=False,
    add_ancillary_variables=False,
):
    """
    Convert a :class:`~iris.cube.Cube` to a :class:`pandas.DataFrame`.

    :attr:`~iris.cube.Cube.dim_coords` and :attr:`~iris.cube.Cube.data` are
    flattened into a long-style :class:`~pandas.DataFrame`.  Other
    :attr:`~iris.cube.Cube.aux_coords`, :attr:`~iris.cube.Cube.aux_coords` and :attr:`~iris.cube.Cube.attributes`
    may be optionally added as additional :class:`~pandas.DataFrame` columns.

    Parameters
    ----------
    cube: :class:`~iris.cube.Cube`
        The :class:`~iris.cube.Cube` to be converted to a :class:`pandas.DataFrame`.
    copy : bool, default=True
        Whether the :class:`pandas.DataFrame` is a copy of the the Cube
        :attr:`~iris.cube.Cube.data`. This option is provided to help with memory
        size concerns.
    add_aux_coords : bool, default=False
        If True, add all :attr:`~iris.cube.Cube.aux_coords` (including scalar
        coordinates) to the returned :class:`pandas.DataFrame`.
    add_cell_measures : bool, default=False
        If True, add :attr:`~iris.cube.Cube.cell_measures` to the returned
        :class:`pandas.DataFrame`.
    add_ancillary_variables: bool, default=False
        If True, add :attr:`~iris.cube.Cube.ancillary_variables` to the returned
        :class:`pandas.DataFrame`.

    Returns
    -------
    :class:`~pandas.DataFrame`
        A :class:`~pandas.DataFrame` with :class:`~iris.cube.Cube` dimensions
        forming a :class:`~pandas.MultiIndex`

    Warnings
    --------
    #. This documentation is for the new ``as_data_frame()`` behaviour, which
       is **currently opt-in** to preserve backwards compatibility. The default
       legacy behaviour is documented in pre-``v3.4`` documentation (summary:
       limited to 2-dimensional :class:`~iris.cube.Cube`\\ s, with only the
       :attr:`~iris.cube.Cube.data` and :attr:`~iris.cube.Cube.dim_coords`
       being added). The legacy behaviour will be removed in a future version
       of Iris, so please opt-in to the new behaviour at your earliest
       convenience, via :class:`iris.Future`:

           >>> iris.FUTURE.pandas_ndim = True

       **Breaking change:** to enable the improvements, the new opt-in
       behaviour flattens multi-dimensional data into a single
       :class:`~pandas.DataFrame` column (the legacy behaviour preserves 2
       dimensions via rows and columns).

       |

    #. Where the :class:`~iris.cube.Cube` contains masked values, these become
       :data:`numpy.nan` in the returned :class:`~pandas.DataFrame`.

    Notes
    -----
    :class:`dask.dataframe.DataFrame`\\ s are not supported.

    A :class:`~pandas.MultiIndex` :class:`~pandas.DataFrame` is returned by default.
    Use the :meth:`~pandas.DataFrame.reset_index` to return a
    :class:`~pandas.DataFrame` without :class:`~pandas.MultiIndex` levels. Use
    'inplace=True` to preserve memory object reference.

    :class:`~iris.cube.Cube` data `dtype` is preserved.

    Examples
    --------
    >>> import iris
    >>> from iris.pandas import as_data_frame
    >>> import pandas as pd
    >>> pd.set_option('display.width', 1000)
    >>> pd.set_option('display.max_columns', 1000)

    Convert a simple :class:`~iris.cube.Cube`:

    >>> path = iris.sample_data_path('ostia_monthly.nc')
    >>> cube = iris.load_cube(path)
    >>> df = as_data_frame(cube)
    >>> print(df)
    ... # doctest: +NORMALIZE_WHITESPACE
                                              surface_temperature
    time                latitude  longitude
    2006-04-16 00:00:00 -4.999992 0.000000             301.659271
                                  0.833333             301.785004
                                  1.666667             301.820984
                                  2.500000             301.865234
                                  3.333333             301.926819
    ...                                                       ...
    2010-09-16 00:00:00  4.444450 355.833313           298.779938
                                  356.666656           298.913147
                                  357.500000                  NaN
                                  358.333313                  NaN
                                  359.166656           298.995148
    <BLANKLINE>
    [419904 rows x 1 columns]

    Using ``add_aux_coords=True`` maps :class:`~iris.coords.AuxCoord` and scalar
    coordinate information to the :class:`~pandas.DataFrame`:

    >>> df = as_data_frame(cube, add_aux_coords=True)
    >>> print(df)
    ... # doctest: +NORMALIZE_WHITESPACE
                                              surface_temperature  forecast_period forecast_reference_time
    time                latitude  longitude
    2006-04-16 00:00:00 -4.999992 0.000000             301.659271                0     2006-04-16 12:00:00
                                  0.833333             301.785004                0     2006-04-16 12:00:00
                                  1.666667             301.820984                0     2006-04-16 12:00:00
                                  2.500000             301.865234                0     2006-04-16 12:00:00
                                  3.333333             301.926819                0     2006-04-16 12:00:00
    ...                                                       ...              ...                     ...
    2010-09-16 00:00:00  4.444450 355.833313           298.779938                0     2010-09-16 12:00:00
                                  356.666656           298.913147                0     2010-09-16 12:00:00
                                  357.500000                  NaN                0     2010-09-16 12:00:00
                                  358.333313                  NaN                0     2010-09-16 12:00:00
                                  359.166656           298.995148                0     2010-09-16 12:00:00
    <BLANKLINE>
    [419904 rows x 3 columns]

    To add netCDF global attribution information to the :class:`~pandas.DataFrame`,
    add a column directly to the :class:`~pandas.DataFrame`:

    >>> df['STASH'] = str(cube.attributes['STASH'])
    >>> print(df)
    ... # doctest: +NORMALIZE_WHITESPACE
                                              surface_temperature  forecast_period forecast_reference_time       STASH
    time                latitude  longitude
    2006-04-16 00:00:00 -4.999992 0.000000             301.659271                0     2006-04-16 12:00:00  m01s00i024
                                  0.833333             301.785004                0     2006-04-16 12:00:00  m01s00i024
                                  1.666667             301.820984                0     2006-04-16 12:00:00  m01s00i024
                                  2.500000             301.865234                0     2006-04-16 12:00:00  m01s00i024
                                  3.333333             301.926819                0     2006-04-16 12:00:00  m01s00i024
    ...                                                       ...              ...                     ...         ...
    2010-09-16 00:00:00  4.444450 355.833313           298.779938                0     2010-09-16 12:00:00  m01s00i024
                                  356.666656           298.913147                0     2010-09-16 12:00:00  m01s00i024
                                  357.500000                  NaN                0     2010-09-16 12:00:00  m01s00i024
                                  358.333313                  NaN                0     2010-09-16 12:00:00  m01s00i024
                                  359.166656           298.995148                0     2010-09-16 12:00:00  m01s00i024
    <BLANKLINE>
    [419904 rows x 4 columns]

    To return a :class:`~pandas.DataFrame` without a :class:`~pandas.MultiIndex`
    use :meth:`~pandas.DataFrame.reset_index`. Optionally use `inplace=True` keyword
    to modify the DataFrame rather than creating a new one:

    >>> df.reset_index(inplace=True)
    >>> print(df)
    ... # doctest: +NORMALIZE_WHITESPACE
                           time  latitude   longitude  surface_temperature  forecast_period forecast_reference_time       STASH
    0       2006-04-16 00:00:00 -4.999992    0.000000           301.659271                0     2006-04-16 12:00:00  m01s00i024
    1       2006-04-16 00:00:00 -4.999992    0.833333           301.785004                0     2006-04-16 12:00:00  m01s00i024
    2       2006-04-16 00:00:00 -4.999992    1.666667           301.820984                0     2006-04-16 12:00:00  m01s00i024
    3       2006-04-16 00:00:00 -4.999992    2.500000           301.865234                0     2006-04-16 12:00:00  m01s00i024
    4       2006-04-16 00:00:00 -4.999992    3.333333           301.926819                0     2006-04-16 12:00:00  m01s00i024
                         ...       ...         ...                  ...              ...                     ...         ...
    419899  2010-09-16 00:00:00  4.444450  355.833313           298.779938                0     2010-09-16 12:00:00  m01s00i024
    419900  2010-09-16 00:00:00  4.444450  356.666656           298.913147                0     2010-09-16 12:00:00  m01s00i024
    419901  2010-09-16 00:00:00  4.444450  357.500000                  NaN                0     2010-09-16 12:00:00  m01s00i024
    419902  2010-09-16 00:00:00  4.444450  358.333313                  NaN                0     2010-09-16 12:00:00  m01s00i024
    419903  2010-09-16 00:00:00  4.444450  359.166656           298.995148                0     2010-09-16 12:00:00  m01s00i024
    <BLANKLINE>
    [419904 rows x 7 columns]

    To retrieve a :class:`~pandas.Series` from `df` :class:`~pandas.DataFrame`,
    subselect a column:

    >>> df['surface_temperature']
    0         301.659271
    1         301.785004
    2         301.820984
    3         301.865234
    4         301.926819
                ...
    419899    298.779938
    419900    298.913147
    419901           NaN
    419902           NaN
    419903    298.995148
    Name: surface_temperature, Length: 419904, dtype: float32

    Notes
    ------
    Since this function converts to/from a Pandas object, laziness will not be preserved.

    """

    def merge_metadata(meta_var_list):
        """Add auxiliary cube metadata to the DataFrame"""
        nonlocal data_frame
        for meta_var_name, meta_var_index, meta_var in meta_var_list:
            if not meta_var_index:
                # Broadcast any meta var informtation without an associated
                # dimension over the whole DataFrame
                data_frame[meta_var_name] = meta_var.squeeze()
            else:
                meta_df = pandas.DataFrame(
                    meta_var.ravel(),
                    columns=[meta_var_name],
                    index=pandas.MultiIndex.from_product(
                        [coords[i] for i in meta_var_index],
                        names=[coord_names[i] for i in meta_var_index],
                    ),
                )
                # Merge to main data frame
                data_frame = pandas.merge(
                    data_frame,
                    meta_df,
                    left_index=True,
                    right_index=True,
                    sort=False,
                )
        return data_frame

    if iris.FUTURE.pandas_ndim:
        # Checks
        if not isinstance(cube, iris.cube.Cube):
            raise TypeError(
                f"Expected input to be iris.cube.Cube instance, got: {type(cube)}"
            )
        if copy:
            data = cube.data.copy()
        else:
            data = cube.data
        if ma.isMaskedArray(data):
            if not copy:
                raise ValueError("Masked arrays must always be copied.")
            data = data.astype("f").filled(np.nan)

        # Extract dim coord information: separate lists for dim names and dim values
        coord_names, coords = _make_dim_coord_list(cube)
        # Make base DataFrame
        index = pandas.MultiIndex.from_product(coords, names=coord_names)
        data_frame = pandas.DataFrame(
            data.ravel(), columns=[cube.name()], index=index
        )

        if add_aux_coords:
            data_frame = merge_metadata(_make_aux_coord_list(cube))
        if add_ancillary_variables:
            data_frame = merge_metadata(_make_ancillary_variables_list(cube))
        if add_cell_measures:
            data_frame = merge_metadata(_make_cell_measures_list(cube))

        if copy:
            result = data_frame.reorder_levels(coord_names).sort_index()
        else:
            data_frame.reorder_levels(coord_names).sort_index(inplace=True)
            result = data_frame

    else:
        message = (
            "You are using legacy 2-dimensional behaviour in"
            "'iris.pandas.as_data_frame()'. This will be removed in a future"
            "version of Iris. Please opt-in to the improved "
            "n-dimensional behaviour at your earliest convenience by setting: "
            "'iris.FUTURE.pandas_ndim = True'. More info is in the "
            "documentation."
        )
        warnings.warn(message, FutureWarning)

        # The legacy behaviour.
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

        result = data_frame

    return result
