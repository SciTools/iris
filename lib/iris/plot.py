# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Iris-specific extensions to matplotlib, mimicking the :mod:`matplotlib.pyplot`
interface.

See also: :ref:`matplotlib <matplotlib:users-guide-index>`.

"""

import collections
import datetime

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import cartopy.mpl.geoaxes
import cftime
import matplotlib.axes
import matplotlib.collections as mpl_collections
import matplotlib.dates as mpl_dates
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import matplotlib.transforms as mpl_transforms
import numpy as np
import numpy.ma as ma

import iris.analysis.cartography as cartography
import iris.coord_systems
import iris.coords
import iris.cube
from iris.exceptions import IrisError

# Importing iris.palette to register the brewer palettes.
import iris.palette
from iris.util import _meshgrid

# Cynthia Brewer citation text.
BREWER_CITE = "Colours based on ColorBrewer.org"

PlotDefn = collections.namedtuple("PlotDefn", ("coords", "transpose"))


def _get_plot_defn_custom_coords_picked(cube, coords, mode, ndims=2):
    def names(coords):
        result = []
        for coord in coords:
            if isinstance(coord, int):
                result.append("dim={}".format(coord))
            else:
                result.append(coord.name())
        return ", ".join(result)

    def as_coord(coord):
        if isinstance(coord, int):
            # Pass through valid dimension indexes.
            if coord >= ndims:
                emsg = (
                    "The data dimension ({}) is out of range for "
                    "the dimensionality of the required plot ({})"
                )
                raise IndexError(emsg.format(coord, ndims))
        else:
            coord = cube.coord(coord)
        return coord

    coords = list(map(as_coord, coords))

    # Check that we were given the right number of coordinates/dimensions.
    if len(coords) != ndims:
        raise ValueError(
            "The list of coordinates given (%s) should have the"
            " same length (%s) as the dimensionality of the"
            " required plot (%s)" % (names(coords), len(coords), ndims)
        )

    # Check which dimensions are spanned by each coordinate.
    def get_span(coord):
        if isinstance(coord, int):
            span = set([coord])
        else:
            span = set(cube.coord_dims(coord))
        return span

    spans = list(map(get_span, coords))
    for span, coord in zip(spans, coords):
        if not span:
            msg = "The coordinate {!r} doesn't span a data dimension."
            raise ValueError(msg.format(coord.name()))
        if mode == iris.coords.BOUND_MODE and len(span) not in [1, 2]:
            raise ValueError(
                "The coordinate {!r} has {} dimensions."
                "Cell-based plotting is only supported for"
                "coordinates with one or two dimensions.".format(
                    coord.name(), len(span)
                )
            )

    # Check the combination of coordinates spans enough (ndims) data
    # dimensions.
    total_span = set().union(*spans)
    if len(total_span) != ndims:
        raise ValueError(
            "The given coordinates ({}) don't span the {} data"
            " dimensions.".format(names(coords), ndims)
        )

    # If we have 2-dimensional data, and one or more 1-dimensional
    # coordinates, check if we need to transpose.
    transpose = False
    if ndims == 2 and min(map(len, spans)) == 1:
        for i, span in enumerate(spans):
            if len(span) == 1:
                if list(span)[0] == i:
                    transpose = True
                    break

    # Note the use of `reversed` to convert from the X-then-Y
    # convention of the end-user API to the V-then-U convention used by
    # the plotting routines.
    plot_coords = list(reversed(coords))
    return PlotDefn(plot_coords, transpose)


def _valid_bound_dim_coord(coord):
    result = None
    if coord and coord.ndim == 1 and coord.nbounds:
        result = coord
    return result


def _get_plot_defn(cube, mode, ndims=2):
    """
    Return data and plot-axis coords given a cube & a mode of either
    POINT_MODE or BOUND_MODE.

    """
    if cube.ndim != ndims:
        msg = "Cube must be %s-dimensional. Got %s dimensions."
        raise ValueError(msg % (ndims, cube.ndim))

    # Start by taking the DimCoords from each dimension.
    coords = [None] * ndims
    for dim_coord in cube.dim_coords:
        dim = cube.coord_dims(dim_coord)[0]
        coords[dim] = dim_coord

    # When appropriate, restrict to 1D with bounds.
    if mode == iris.coords.BOUND_MODE:
        coords = list(map(_valid_bound_dim_coord, coords))

    def guess_axis(coord):
        axis = None
        if coord is not None:
            axis = iris.util.guess_coord_axis(coord)
        return axis

    # Allow DimCoords in aux_coords to fill in for missing dim_coords.
    for dim, coord in enumerate(coords):
        if coord is None:
            aux_coords = cube.coords(dimensions=dim)
            aux_coords = [
                coord
                for coord in aux_coords
                if isinstance(coord, iris.coords.DimCoord)
            ]
            if aux_coords:
                aux_coords.sort(key=lambda coord: coord.metadata)
                coords[dim] = aux_coords[0]

    # If plotting a 2 dimensional plot, check for 2d coordinates
    if ndims == 2:
        missing_dims = [
            dim for dim, coord in enumerate(coords) if coord is None
        ]
        if missing_dims:
            # Note that this only picks up coordinates that span the dims
            two_dim_coords = cube.coords(dimensions=missing_dims)
            two_dim_coords = [
                coord for coord in two_dim_coords if coord.ndim == 2
            ]
            if len(two_dim_coords) >= 2:
                two_dim_coords.sort(key=lambda coord: coord.metadata)
                coords = two_dim_coords[:2]

    if mode == iris.coords.POINT_MODE:
        # Allow multi-dimensional aux_coords to override the dim_coords
        # along the Z axis. This results in a preference for using the
        # derived altitude over model_level_number or level_height.
        # Limit to Z axis to avoid preferring latitude over grid_latitude etc.
        axes = list(map(guess_axis, coords))
        axis = "Z"
        if axis in axes:
            for coord in cube.coords(dim_coords=False):
                if (
                    max(coord.shape) > 1
                    and iris.util.guess_coord_axis(coord) == axis
                ):
                    coords[axes.index(axis)] = coord

    # Re-order the coordinates to achieve the preferred
    # horizontal/vertical associations. If we can't associate
    # an axis to order the coordinates, fall back to using the cube dimension
    # followed by the name of the coordinate.
    def sort_key(coord):
        order = {"X": 2, "T": 1, "Y": -1, "Z": -2}
        axis = guess_axis(coord)
        return (
            order.get(axis, 0),
            coords.index(coord),
            coord and coord.name(),
        )

    sorted_coords = sorted(coords, key=sort_key)

    transpose = sorted_coords != coords
    return PlotDefn(sorted_coords, transpose)


def _can_draw_map(coords):
    std_names = [
        c and c.standard_name
        for c in coords
        if isinstance(c, iris.coords.Coord)
    ]
    valid_std_names = [
        ["latitude", "longitude"],
        ["grid_latitude", "grid_longitude"],
        ["projection_y_coordinate", "projection_x_coordinate"],
    ]
    return std_names in valid_std_names


def _broadcast_2d(u, v):
    # Matplotlib needs the U and V coordinates to have the same
    # dimensionality (either both 1D, or both 2D). So we simply
    # broadcast both to 2D to be on the safe side.
    u = np.atleast_2d(u)
    v = np.atleast_2d(v.T).T
    u, v = np.broadcast_arrays(u, v)
    return u, v


def _string_coord_axis_tick_labels(string_axes, axes=None):
    """Apply tick labels for string coordinates."""

    ax = axes if axes else plt.gca()
    for axis, ticks in string_axes.items():
        # Define a tick formatter. This will assign a label to all ticks
        # located precisely on  an integer in range(len(ticks)) and assign
        # an empty string to any other ticks.
        def ticker_func(tick_location, _):
            tick_locations = range(len(ticks))
            labels = ticks
            label_dict = dict(zip(tick_locations, labels))
            label = label_dict.get(tick_location, "")
            return label

        formatter = mpl_ticker.FuncFormatter(ticker_func)
        locator = mpl_ticker.MaxNLocator(integer=True)
        this_axis = getattr(ax, axis)
        this_axis.set_major_formatter(formatter)
        this_axis.set_major_locator(locator)


def _invert_yaxis(v_coord, axes=None):
    """
    Inverts the y-axis of the current plot based on conditions:

        * If the y-axis is already inverted we don't want to re-invert it.
        * If v_coord is None then it will not have any attributes.
        * If neither of the above are true then invert y if v_coord has
          attribute 'positive' set to 'down'.

    Args:

        * v_coord - the coord to be plotted on the y-axis

    """
    axes = axes if axes else plt.gca()
    yaxis_is_inverted = axes.yaxis_inverted()
    if not yaxis_is_inverted and isinstance(v_coord, iris.coords.Coord):
        attr_pve = v_coord.attributes.get("positive")
        if attr_pve is not None and attr_pve.lower() == "down":
            axes.invert_yaxis()


def _check_bounds_contiguity_and_mask(coord, data, atol=None, rtol=None):
    """
    Checks that any discontiguities in the bounds of the given coordinate only
    occur where the data is masked.

    Where a discontinuity occurs the grid created for plotting will not be
    correct. This does not matter if the data is masked in that location as
    this is not plotted.

    If a discontiguity occurs where the data is *not* masked, an error is
    raised.

    Args:
        coord: (iris.coord.Coord)
            Coordinate the bounds of which will be checked for contiguity
        data: (array)
            Data of the the cube we are plotting
        atol:
            Absolute tolerance when checking the contiguity. Defaults to None.
            If an absolute tolerance is not set, 1D coords are not checked (so
            as to not introduce a breaking change without a major release) but
            2D coords are always checked, by calling
            :meth:`iris.coords.Coord._discontiguity_in_bounds` with its default
            tolerance.

    """
    kwargs = {}
    data_is_masked = hasattr(data, "mask")
    if data_is_masked:
        # When checking the location of the discontiguities, we check against
        # the opposite of the mask, which is True where data exists.
        mask_invert = np.logical_not(data.mask)

    if coord.ndim == 1:
        # 1D coords are only checked if an absolute tolerance is set, to avoid
        # introducing a breaking change.
        if atol:
            contiguous, diffs = coord._discontiguity_in_bounds(atol=atol)

            if not contiguous and data_is_masked:
                not_masked_at_discontiguity = np.any(
                    np.logical_and(mask_invert[:-1], diffs)
                )
        else:
            return

    elif coord.ndim == 2:
        if atol:
            kwargs["atol"] = atol
        if rtol:
            kwargs["rtol"] = rtol
        contiguous, diffs = coord._discontiguity_in_bounds(**kwargs)

        if not contiguous and data_is_masked:
            diffs_along_x, diffs_along_y = diffs

            # Check along both dimensions that any discontiguous
            # points are correctly masked.
            not_masked_at_discontiguity_along_x = np.any(
                np.logical_and(mask_invert[:, :-1], diffs_along_x)
            )

            not_masked_at_discontiguity_along_y = np.any(
                np.logical_and(mask_invert[:-1], diffs_along_y)
            )

            not_masked_at_discontiguity = (
                not_masked_at_discontiguity_along_x
                or not_masked_at_discontiguity_along_y
            )

    # If any discontiguity occurs where the data is not masked the grid will be
    # created incorrectly, so raise an error.
    if not contiguous:
        if not data_is_masked:
            raise ValueError(
                "The bounds of the {} coordinate are not "
                "contiguous. Not able to create a suitable grid"
                "to plot. You can use "
                "iris.util.find_discontiguities() to identify "
                "discontiguities in your x and y coordinate "
                "bounds arrays.".format(coord.name())
            )
        if not_masked_at_discontiguity:
            raise ValueError(
                "The bounds of the {} coordinate are not "
                "contiguous and data is not masked where the "
                "discontiguity occurs. Not able to create a "
                "suitable grid to plot. You can use "
                "iris.util.find_discontiguities() to identify "
                "discontiguities in your x and y coordinate "
                "bounds arrays, and then mask them with "
                "iris.util.mask_cube()"
                "".format(coord.name())
            )


def _draw_2d_from_bounds(draw_method_name, cube, *args, **kwargs):
    # NB. In the interests of clarity we use "u" and "v" to refer to the
    # horizontal and vertical axes on the matplotlib plot.
    mode = iris.coords.BOUND_MODE
    # Get & remove the coords entry from kwargs.
    coords = kwargs.pop("coords", None)
    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(
            cube, coords, mode, ndims=2
        )
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)

    contig_tol = kwargs.pop("contiguity_tolerance", None)

    for coord in plot_defn.coords:
        if hasattr(coord, "has_bounds") and coord.has_bounds():
            _check_bounds_contiguity_and_mask(
                coord, data=cube.data, atol=contig_tol
            )

    if _can_draw_map(plot_defn.coords):
        result = _map_common(
            draw_method_name,
            None,
            iris.coords.BOUND_MODE,
            cube,
            plot_defn,
            *args,
            **kwargs,
        )
    else:
        # Obtain data array.
        data = cube.data
        if plot_defn.transpose:
            data = data.T

        # Obtain U and V coordinates
        v_coord, u_coord = plot_defn.coords

        # Track numpy arrays to use for the actual plotting.
        plot_arrays = []

        # Map axis name to associated values.
        string_axes = {}

        for coord, axis_name, data_dim in zip(
            [u_coord, v_coord], ["xaxis", "yaxis"], [1, 0]
        ):
            if coord is None:
                values = np.arange(data.shape[data_dim] + 1)
            elif isinstance(coord, int):
                dim = 1 - coord if plot_defn.transpose else coord
                values = np.arange(data.shape[dim] + 1)
            else:
                if coord.points.dtype.char in "SU":
                    if coord.points.ndim != 1:
                        msg = "Coord {!r} must be one-dimensional."
                        raise ValueError(msg.format(coord))
                    if coord.bounds is not None:
                        msg = "Cannot plot bounded string coordinate."
                        raise ValueError(msg)
                    string_axes[axis_name] = coord.points
                    values = np.arange(data.shape[data_dim] + 1) - 0.5
                else:
                    values = coord.contiguous_bounds()
                    values = _fixup_dates(coord, values)
                    if values.dtype == np.dtype(object) and isinstance(
                        values[0], datetime.datetime
                    ):
                        values = mpl_dates.date2num(values)

            plot_arrays.append(values)

        u, v = plot_arrays

        # If the data is transposed, 2D coordinates will also need to be
        # transposed.
        if plot_defn.transpose is True:
            u, v = [coord.T if coord.ndim == 2 else coord for coord in [u, v]]

        if u.ndim == v.ndim == 1:
            u, v = _broadcast_2d(u, v)

        axes = kwargs.pop("axes", None)
        draw_method = getattr(axes if axes else plt, draw_method_name)
        result = draw_method(u, v, data, *args, **kwargs)

        # Apply tick labels for string coordinates.
        _string_coord_axis_tick_labels(string_axes, axes)

        # Invert y-axis if necessary.
        _invert_yaxis(v_coord, axes)

    return result


def _draw_2d_from_points(draw_method_name, arg_func, cube, *args, **kwargs):
    # NB. In the interests of clarity we use "u" and "v" to refer to the
    # horizontal and vertical axes on the matplotlib plot.
    mode = iris.coords.POINT_MODE
    # Get & remove the coords entry from kwargs.
    coords = kwargs.pop("coords", None)
    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode)
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)

    if _can_draw_map(plot_defn.coords):
        result = _map_common(
            draw_method_name,
            arg_func,
            iris.coords.POINT_MODE,
            cube,
            plot_defn,
            *args,
            **kwargs,
        )
    else:
        # Obtain data array.
        data = cube.data
        if plot_defn.transpose:
            data = data.T
            # Also transpose the scatter marker color array,
            # as now mpl 2.x does not do this for free.
            if draw_method_name == "scatter" and "c" in kwargs:
                c = kwargs["c"]
                if hasattr(c, "T") and cube.data.shape == c.shape:
                    kwargs["c"] = c.T

        # Obtain U and V coordinates
        v_coord, u_coord = plot_defn.coords
        if u_coord is None:
            u = np.arange(data.shape[1])
        elif isinstance(u_coord, int):
            dim = 1 - u_coord if plot_defn.transpose else u_coord
            u = np.arange(data.shape[dim])
        else:
            u = u_coord.points
            u = _fixup_dates(u_coord, u)

        if v_coord is None:
            v = np.arange(data.shape[0])
        elif isinstance(v_coord, int):
            dim = 1 - v_coord if plot_defn.transpose else v_coord
            v = np.arange(data.shape[dim])
        else:
            v = v_coord.points
            v = _fixup_dates(v_coord, v)

        if plot_defn.transpose:
            u = u.T
            v = v.T

        # Track numpy arrays to use for the actual plotting.
        plot_arrays = []

        # Map axis name to associated values.
        string_axes = {}

        for values, axis_name in zip([u, v], ["xaxis", "yaxis"]):
            # Replace any string coordinates with "index" coordinates.
            if values.dtype.char in "SU":
                if values.ndim != 1:
                    raise ValueError(
                        "Multi-dimensional string coordinates "
                        "not supported."
                    )
                plot_arrays.append(np.arange(values.size))
                string_axes[axis_name] = values
            elif values.dtype == np.dtype(object) and isinstance(
                values[0], datetime.datetime
            ):
                plot_arrays.append(mpl_dates.date2num(values))
            else:
                plot_arrays.append(values)

        u, v = plot_arrays
        u, v = _broadcast_2d(u, v)

        axes = kwargs.pop("axes", None)
        draw_method = getattr(axes if axes else plt, draw_method_name)
        if arg_func is not None:
            args, kwargs = arg_func(u, v, data, *args, **kwargs)
            result = draw_method(*args, **kwargs)
        else:
            result = draw_method(u, v, data, *args, **kwargs)

        # Apply tick labels for string coordinates.
        _string_coord_axis_tick_labels(string_axes, axes)

        # Invert y-axis if necessary.
        _invert_yaxis(v_coord, axes)

    return result


def _fixup_dates(coord, values):
    if coord.units.calendar is not None and values.ndim == 1:
        # Convert coordinate values into tuples of
        # (year, month, day, hour, min, sec)
        dates = [coord.units.num2date(val).timetuple()[0:6] for val in values]
        if coord.units.calendar == "gregorian":
            r = [datetime.datetime(*date) for date in dates]
        else:
            try:
                import nc_time_axis  # noqa: F401
            except ImportError:
                msg = (
                    "Cannot plot against time in a non-gregorian "
                    'calendar, because "nc_time_axis" is not available :  '
                    "Install the package from "
                    "https://github.com/SciTools/nc-time-axis to enable "
                    "this usage."
                )
                raise IrisError(msg)

            r = [
                cftime.datetime(*date, calendar=coord.units.calendar)
                for date in dates
            ]

        values = np.empty(len(r), dtype=object)
        values[:] = r
    return values


def _data_from_coord_or_cube(c):
    if isinstance(c, iris.cube.Cube):
        data = c.data
    elif isinstance(c, iris.coords.Coord):
        data = _fixup_dates(c, c.points)
    else:
        raise TypeError("Plot arguments must be cubes or coordinates.")
    return data


def _uv_from_u_object_v_object(u_object, v_object):
    ndim_msg = "Cube or coordinate must be 1-dimensional. Got {} dimensions."
    if u_object is not None and u_object.ndim > 1:
        raise ValueError(ndim_msg.format(u_object.ndim))
    if v_object.ndim > 1:
        raise ValueError(ndim_msg.format(v_object.ndim))
    v = _data_from_coord_or_cube(v_object)
    if u_object is None:
        u = np.arange(v.shape[0])
    else:
        u = _data_from_coord_or_cube(u_object)
    return u, v


def _u_object_from_v_object(v_object):
    u_object = None
    if isinstance(v_object, iris.cube.Cube):
        plot_defn = _get_plot_defn(v_object, iris.coords.POINT_MODE, ndims=1)
        (u_object,) = plot_defn.coords
    return u_object


def _get_plot_objects(args):
    if len(args) > 1 and isinstance(
        args[1], (iris.cube.Cube, iris.coords.Coord)
    ):
        # two arguments
        u_object, v_object = args[:2]
        u, v = _uv_from_u_object_v_object(u_object, v_object)
        args = args[2:]
        if u.size != v.size:
            msg = (
                "The x and y-axis objects are not compatible. They should "
                "have equal sizes but got ({}: {}) and ({}: {})."
            )
            raise ValueError(
                msg.format(u_object.name(), u.size, v_object.name(), v.size)
            )
    else:
        # single argument
        v_object = args[0]
        u_object = _u_object_from_v_object(v_object)

        u, v = _uv_from_u_object_v_object(u_object, args[0])

        # If a single cube argument, and the associated dimension coordinate
        # is vertical-like, put the coordinate on the y axis, and the data o
        # the x.
        if (
            isinstance(v_object, iris.cube.Cube)
            and isinstance(u_object, iris.coords.Coord)
            and iris.util.guess_coord_axis(u_object) == "Z"
        ):
            u_object, v_object = v_object, u_object
            u, v = v, u

        args = args[1:]
    return u_object, v_object, u, v, args


def _get_geodesic_params(globe):
    # Derive the semimajor axis and flattening values for a given globe from
    # its attributes. If the values are under specified, raise a ValueError
    flattening = globe.flattening
    semimajor = globe.semimajor_axis
    try:
        if semimajor is None:
            # Has semiminor or raises error
            if flattening is None:
                # Has inverse flattening or raises error
                flattening = 1.0 / globe.inverse_flattening
            semimajor = globe.semiminor_axis / (1.0 - flattening)
        elif flattening is None:
            if globe.semiminor_axis is not None:
                flattening = (semimajor - globe.semiminor_axis) / float(
                    semimajor
                )
            else:
                # Has inverse flattening or raises error
                flattening = 1.0 / globe.inverse_flattening
    except TypeError:
        # One of the required attributes was None
        raise ValueError("The globe was underspecified.")

    return semimajor, flattening


def _shift_plot_sections(u_object, u, v):
    """
    Shifts subsections of u by multiples of 360 degrees within ranges
    defined by the points where the line should cross over the 0/360 degree
    longitude boundary.

    e.g. [ 300, 100, 200, 300, 100, 300 ] => [ 300, 460, 560, 660, 820, 660 ]

    """
    # Convert coordinates to true lat-lon
    src_crs = (
        u_object.coord_system.as_cartopy_crs()
        if u_object.coord_system is not None
        else ccrs.Geodetic()
    )
    tgt_crs = ccrs.Geodetic(globe=src_crs.globe)
    tgt_proj = ccrs.PlateCarree(globe=src_crs.globe)

    points = tgt_crs.transform_points(src_crs, u, v)
    startpoints = points[:-1, :2]
    endpoints = points[1:, :2]
    proj_x, proj_y, _ = tgt_proj.transform_points(src_crs, u, v).T

    # Calculate the inverse geodesic for each pair of points in turn, and
    # convert the start point's azimuth into a vector in the source coordinate
    # system.
    try:
        radius, flattening = _get_geodesic_params(src_crs.globe)
        geodesic = Geodesic(radius, flattening)
    except ValueError:
        geodesic = Geodesic()
    dists, azms, _ = geodesic.inverse(startpoints, endpoints).T
    azms_lon = np.sin(np.deg2rad(azms))
    azms_lat = np.cos(np.deg2rad(azms))
    azms_u, _ = src_crs.transform_vectors(
        tgt_proj, proj_x[:-1], proj_y[:-1], azms_lon, azms_lat
    )

    # Use the grid longitude values and the geodesic azimuth to determine
    # the points where the line should cross the 0/360 degree boundary, and
    # in which direction
    lwraps = np.logical_and(u[1:] > u[:-1], azms_u < 0)
    rwraps = np.logical_and(u[1:] < u[:-1], azms_u > 0)
    shifts = np.where(rwraps, 1, 0) - np.where(lwraps, 1, 0)
    shift_vals = shifts.cumsum() * u_object.units.modulus
    new_u = np.empty_like(u)
    new_u[0] = u[0]
    new_u[1:] = u[1:] + shift_vals
    return new_u


def _draw_1d_from_points(draw_method_name, arg_func, *args, **kwargs):
    # NB. In the interests of clarity we use "u" to refer to the horizontal
    # axes on the matplotlib plot and "v" for the vertical axes.

    # retrieve the objects that are plotted on the horizontal and vertical
    # axes (cubes or coordinates) and their respective values, along with the
    # argument tuple with these objects removed
    u_object, v_object, u, v, args = _get_plot_objects(args)

    # Track numpy arrays to use for the actual plotting.
    plot_arrays = []

    # Map axis name to associated values.
    string_axes = {}

    for values, axis_name in zip([u, v], ["xaxis", "yaxis"]):
        # Replace any string coordinates with "index" coordinates.
        if values.dtype.char in "SU":
            if values.ndim != 1:
                msg = "Multi-dimensional string coordinates are not supported."
                raise ValueError(msg)
            plot_arrays.append(np.arange(values.size))
            string_axes[axis_name] = values
        else:
            plot_arrays.append(values)

    u, v = plot_arrays

    # if both u_object and v_object are coordinates then check if a map
    # should be drawn
    if (
        isinstance(u_object, iris.coords.Coord)
        and isinstance(v_object, iris.coords.Coord)
        and _can_draw_map([v_object, u_object])
    ):
        # Replace non-cartopy subplot/axes with a cartopy alternative and set
        # the transform keyword.
        kwargs = _ensure_cartopy_axes_and_determine_kwargs(
            u_object, v_object, kwargs
        )
        if draw_method_name == "plot" and u_object.standard_name not in (
            "projection_x_coordinate",
            "projection_y_coordinate",
        ):
            u = _shift_plot_sections(u_object, u, v)

    axes = kwargs.pop("axes", None)
    draw_method = getattr(axes if axes else plt, draw_method_name)
    if arg_func is not None:
        args, kwargs = arg_func(u, v, *args, **kwargs)
        result = draw_method(*args, **kwargs)
    else:
        result = draw_method(u, v, *args, **kwargs)

    # Apply tick labels for string coordinates.
    _string_coord_axis_tick_labels(string_axes, axes)

    # Invert y-axis if necessary.
    _invert_yaxis(v_object, axes)

    return result


def _replace_axes_with_cartopy_axes(cartopy_proj):
    """
    Replace non-cartopy subplot/axes with a cartopy alternative
    based on the provided projection. If the current axes are already an
    instance of :class:`cartopy.mpl.geoaxes.GeoAxes` then no action is taken.

    """

    ax = plt.gca()
    if not isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        fig = plt.gcf()
        if isinstance(ax, matplotlib.axes.SubplotBase):
            _ = fig.add_subplot(
                ax.get_subplotspec(),
                projection=cartopy_proj,
                title=ax.get_title(),
                xlabel=ax.get_xlabel(),
                ylabel=ax.get_ylabel(),
            )
        else:
            _ = fig.add_axes(
                projection=cartopy_proj,
                title=ax.get_title(),
                xlabel=ax.get_xlabel(),
                ylabel=ax.get_ylabel(),
            )

        # delete the axes which didn't have a cartopy projection
        fig.delaxes(ax)


def _ensure_cartopy_axes_and_determine_kwargs(x_coord, y_coord, kwargs):
    """
    Replace the current non-cartopy axes with
    :class:`cartopy.mpl.geoaxes.GeoAxes` and return the appropriate kwargs dict
    based on the provided coordinates and kwargs.

    """
    # Determine projection.
    if x_coord.coord_system != y_coord.coord_system:
        raise ValueError(
            "The X and Y coordinates must have equal coordinate" " systems."
        )
    cs = x_coord.coord_system
    if cs is not None:
        cartopy_proj = cs.as_cartopy_projection()
    else:
        cartopy_proj = ccrs.PlateCarree()

    # Ensure the current axes are a cartopy.mpl.geoaxes.GeoAxes instance.
    axes = kwargs.get("axes")
    if axes is None:
        if (
            isinstance(cs, iris.coord_systems.RotatedGeogCS)
            and x_coord.points.max() > 180
            and x_coord.points.max() < 360
            and x_coord.points.min() > 0
        ):
            # The RotatedGeogCS has 0 - 360 extent, different from the
            # assumptions made by Cartopy: rebase longitudes for the map axes
            # to set the datum longitude to the International Date Line.
            cs_kwargs = cs._ccrs_kwargs()
            cs_kwargs["central_rotated_longitude"] = 180.0
            adapted_cartopy_proj = ccrs.RotatedPole(**cs_kwargs)
            _replace_axes_with_cartopy_axes(adapted_cartopy_proj)
        else:
            _replace_axes_with_cartopy_axes(cartopy_proj)
    elif axes and not isinstance(axes, cartopy.mpl.geoaxes.GeoAxes):
        raise TypeError(
            "The supplied axes instance must be a cartopy " "GeoAxes instance."
        )

    # Set the "from transform" keyword.
    if "transform" in kwargs:
        raise ValueError(
            "The 'transform' keyword is not allowed as it "
            "automatically determined from the coordinate "
            "metadata."
        )
    new_kwargs = kwargs.copy()
    new_kwargs["transform"] = cartopy_proj

    return new_kwargs


def _check_geostationary_coords_and_convert(x, y, kwargs):
    # Geostationary stores projected coordinates as scanning angles (
    # radians), in line with CF definition (this behaviour is unique to
    # Geostationary). Before plotting, must be converted by multiplying by
    # satellite height.
    x, y = (i.copy() for i in (x, y))
    transform = kwargs.get("transform")
    if isinstance(transform, cartopy.crs.Geostationary):
        satellite_height = transform.proj4_params["h"]
        for i in (x, y):
            i *= satellite_height

    return x, y


def _map_common(
    draw_method_name, arg_func, mode, cube, plot_defn, *args, **kwargs
):
    """
    Draw the given cube on a map using its points or bounds.

    "Mode" parameter will switch functionality between POINT or BOUND plotting.


    """
    # Generate 2d x and 2d y grids.
    y_coord, x_coord = plot_defn.coords
    if mode == iris.coords.POINT_MODE:
        if x_coord.ndim == y_coord.ndim == 1:
            x, y = _meshgrid(x_coord.points, y_coord.points)
        elif x_coord.ndim == y_coord.ndim == 2:
            x = x_coord.points
            y = y_coord.points
        else:
            raise ValueError("Expected 1D or 2D XY coords")
    else:
        if not x_coord.ndim == y_coord.ndim == 2:
            try:
                x, y = _meshgrid(
                    x_coord.contiguous_bounds(), y_coord.contiguous_bounds()
                )
            # Exception translation.
            except iris.exceptions.CoordinateMultiDimError:
                raise ValueError(
                    "Expected two 1D coords. Could not get XY"
                    " grid from bounds. X or Y coordinate not"
                    " 1D."
                )
            except ValueError:
                raise ValueError(
                    "Could not get XY grid from bounds. "
                    "X or Y coordinate doesn't have 2 bounds "
                    "per point."
                )
        else:
            x = x_coord.contiguous_bounds()
            y = y_coord.contiguous_bounds()

    # Obtain the data array.
    data = cube.data
    if plot_defn.transpose:
        data = data.T

    # If we are global, then append the first column of data the array to the
    # last (and add 360 degrees) NOTE: if it is found that this block of code
    # is useful in anywhere other than this plotting routine, it may be better
    # placed in the CS.
    if getattr(x_coord, "circular", False):
        _, direction = iris.util.monotonic(
            x_coord.points, return_direction=True
        )
        y = np.append(y, y[:, 0:1], axis=1)
        x = np.append(x, x[:, 0:1] + 360 * direction, axis=1)
        data = ma.concatenate([data, data[:, 0:1]], axis=1)
        if "_v_data" in kwargs:
            v_data = kwargs["_v_data"]
            v_data = ma.concatenate([v_data, v_data[:, 0:1]], axis=1)
            kwargs["_v_data"] = v_data

    # Replace non-cartopy subplot/axes with a cartopy alternative and set the
    # transform keyword.
    kwargs = _ensure_cartopy_axes_and_determine_kwargs(
        x_coord, y_coord, kwargs
    )

    # Make Geostationary coordinates plot-able.
    x, y = _check_geostationary_coords_and_convert(x, y, kwargs)

    if arg_func is not None:
        new_args, kwargs = arg_func(x, y, data, *args, **kwargs)
    else:
        new_args = (x, y, data) + args

    # Draw the contour lines/filled contours.
    axes = kwargs.pop("axes", None)
    plotfn = getattr(axes if axes else plt, draw_method_name)
    return plotfn(*new_args, **kwargs)


def contour(cube, *args, **kwargs):
    """
    Draws contour lines based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the
        plot. The order of the given coordinates indicates which axis
        to use for each, where the first element is the horizontal
        axis of the plot and the second element is the vertical axis
        of the plot.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    See :func:`matplotlib.pyplot.contour` for details of other valid
    keyword arguments.

    """
    result = _draw_2d_from_points("contour", None, cube, *args, **kwargs)
    return result


def contourf(cube, *args, **kwargs):
    """
    Draws filled contours based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    See :func:`matplotlib.pyplot.contourf` for details of other valid
    keyword arguments.

    """
    coords = kwargs.get("coords")
    kwargs.setdefault("antialiased", True)
    result = _draw_2d_from_points("contourf", None, cube, *args, **kwargs)

    # Matplotlib produces visible seams between anti-aliased polygons.
    # But if the polygons are virtually opaque then we can cover the seams
    # by drawing anti-aliased lines *underneath* the polygon joins.

    # Figure out the alpha level for the contour plot
    if result.alpha is None:
        alpha = result.collections[0].get_facecolor()[0][3]
    else:
        alpha = result.alpha
    # If the contours are anti-aliased and mostly opaque then draw lines under
    # the seams.
    if result.antialiased and alpha > 0.95:
        levels = result.levels
        colors = [c[0] for c in result.tcolors]
        if result.extend == "neither":
            levels = levels[1:-1]
            colors = colors[:-1]
        elif result.extend == "min":
            levels = levels[:-1]
            colors = colors[:-1]
        elif result.extend == "max":
            levels = levels[1:]
            colors = colors[:-1]
        else:
            colors = colors[:-1]
        if len(levels) > 0 and np.nanmax(cube.data) > levels[0]:
            # Draw the lines just *below* the polygons to ensure we minimise
            # any boundary shift.
            zorder = result.collections[0].zorder - 0.1
            axes = kwargs.get("axes", None)

            contour(
                cube,
                levels=levels,
                colors=colors,
                antialiased=True,
                zorder=zorder,
                coords=coords,
                axes=axes,
            )
            # Restore the current "image" to 'result' rather than the mappable
            # resulting from the additional call to contour().
            if axes:
                axes._sci(result)
            else:
                plt.sci(result)

    return result


def default_projection(cube):
    """
    Return the primary map projection for the given cube.

    Using the returned projection, one can create a cartopy map with::

        import matplotlib.pyplot as plt
        ax = plt.ax(projection=default_projection(cube))

    """
    # XXX logic seems flawed, but it is what map_setup did...
    cs = cube.coord_system("CoordSystem")
    projection = cs.as_cartopy_projection() if cs else None
    return projection


def default_projection_extent(cube, mode=iris.coords.POINT_MODE):
    """
    Return the cube's extents ``(x0, x1, y0, y1)`` in its default projection.

    Keyword arguments:

    * mode: Either ``iris.coords.POINT_MODE`` or ``iris.coords.BOUND_MODE``
            Triggers whether the extent should be representative of the cell
            points, or the limits of the cell's bounds.
            The default is iris.coords.POINT_MODE.

    """
    extents = cartography._xy_range(cube, mode)
    xlim = extents[0]
    ylim = extents[1]
    return tuple(xlim) + tuple(ylim)


def _fill_orography(cube, coords, mode, vert_plot, horiz_plot, style_args):
    # Find the orography coordinate.
    orography = cube.coord("surface_altitude")

    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(
            cube, coords, mode, ndims=2
        )
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)
    v_coord, u_coord = plot_defn.coords

    # Find which plot coordinate corresponds to the derived altitude, so that
    # we can replace altitude with the surface altitude.
    if v_coord and v_coord.standard_name == "altitude":
        # v is altitude, so plot u and orography with orog in the y direction.
        result = vert_plot(u_coord, orography, style_args)
    elif u_coord and u_coord.standard_name == "altitude":
        # u is altitude, so plot v and orography with orog in the x direction.
        result = horiz_plot(v_coord, orography, style_args)
    else:
        raise ValueError(
            "Plot does not use hybrid height. One of the "
            "coordinates to plot must be altitude, but %s and %s "
            "were given." % (u_coord.name(), v_coord.name())
        )
    return result


def orography_at_bounds(cube, facecolor="#888888", coords=None, axes=None):
    """Plots orography defined at cell boundaries from the given Cube."""

    # XXX Needs contiguous orography corners to work.
    raise NotImplementedError(
        "This operation is temporarily not provided "
        "until coordinates can expose 2d contiguous "
        "bounds (corners)."
    )

    style_args = {"edgecolor": "none", "facecolor": facecolor}

    def vert_plot(u_coord, orography, style_args):
        u = u_coord.contiguous_bounds()
        left = u[:-1]
        height = orography.points
        width = u[1:] - left
        plotfn = axes.bar if axes else plt.bar
        return plotfn(left, height, width, **style_args)

    def horiz_plot(v_coord, orography, style_args):
        v = v_coord.contiguous_bounds()
        bottom = v[:-1]
        width = orography.points
        height = v[1:] - bottom
        plotfn = axes.barh if axes else plt.barh
        return plotfn(bottom, width, height, **style_args)

    return _fill_orography(
        cube, coords, iris.coords.BOUND_MODE, vert_plot, horiz_plot, style_args
    )


def orography_at_points(cube, facecolor="#888888", coords=None, axes=None):
    """Plots orography defined at sample points from the given Cube."""

    style_args = {"facecolor": facecolor}

    def vert_plot(u_coord, orography, style_args):
        x = u_coord.points
        y = orography.points
        plotfn = axes.fill_between if axes else plt.fill_between
        return plotfn(x, y, **style_args)

    def horiz_plot(v_coord, orography, style_args):
        y = v_coord.points
        x = orography.points
        plotfn = axes.fill_betweenx if axes else plt.fill_betweenx
        return plotfn(y, x, **style_args)

    return _fill_orography(
        cube, coords, iris.coords.POINT_MODE, vert_plot, horiz_plot, style_args
    )


def outline(cube, coords=None, color="k", linewidth=None, axes=None):
    """
    Draws cell outlines based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the
        plot. The order of the given coordinates indicates which axis
        to use for each, where the first element is the horizontal
        axis of the plot and the second element is the vertical axis
        of the plot.

    * color: None or mpl color
        The color of the cell outlines. If None, the matplotlibrc setting
        patch.edgecolor is used by default.

    * linewidth: None or number
        The width of the lines showing the cell outlines. If None, the default
        width in patch.linewidth in matplotlibrc is used.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    """
    result = _draw_2d_from_bounds(
        "pcolormesh",
        cube,
        facecolors="none",
        edgecolors=color,
        linewidth=linewidth,
        antialiased=True,
        coords=coords,
        axes=axes,
    )

    # set the _is_stroked property to get a single color grid.
    # See https://github.com/matplotlib/matplotlib/issues/1302
    result._is_stroked = False
    if hasattr(result, "_wrapped_collection_fix"):
        result._wrapped_collection_fix._is_stroked = False
    return result


def pcolor(cube, *args, **kwargs):
    """
    Draws a pseudocolor plot based on the given 2-dimensional Cube.

    The cube must have either two 1-dimensional coordinates or two
    2-dimensional coordinates with contiguous bounds to plot the cube against.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the
        plot. The order of the given coordinates indicates which axis
        to use for each, where the first element is the horizontal
        axis of the plot and the second element is the vertical axis
        of the plot.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    * contiguity_tolerance: float
        The absolute tolerance used when checking for contiguity between the
        bounds of the cells. Defaults to None.

    See :func:`matplotlib.pyplot.pcolor` for details of other valid
    keyword arguments.

    """
    kwargs.setdefault("antialiased", True)
    kwargs.setdefault("snap", False)
    result = _draw_2d_from_bounds("pcolor", cube, *args, **kwargs)
    return result


def pcolormesh(cube, *args, **kwargs):
    """
    Draws a pseudocolor plot based on the given 2-dimensional Cube.

    The cube must have either two 1-dimensional coordinates or two
    2-dimensional coordinates with contiguous bounds to plot against each
    other.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    * contiguity_tolerance: float
        The absolute tolerance used when checking for
        contiguity between the bounds of the cells. Defaults to None.

    See :func:`matplotlib.pyplot.pcolormesh` for details of other
    valid keyword arguments.

    """
    result = _draw_2d_from_bounds("pcolormesh", cube, *args, **kwargs)
    return result


def points(cube, *args, **kwargs):
    """
    Draws sample point positions based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the
        plot. The order of the given coordinates indicates which axis
        to use for each, where the first element is the horizontal
        axis of the plot and the second element is the vertical axis
        of the plot.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    See :func:`matplotlib.pyplot.scatter` for details of other valid
    keyword arguments.

    """

    def _scatter_args(u, v, data, *args, **kwargs):
        return ((u, v) + args, kwargs)

    return _draw_2d_from_points(
        "scatter", _scatter_args, cube, *args, **kwargs
    )


def _vector_component_args(x_points, y_points, u_data, *args, **kwargs):
    """
    Callback from _draw_2d_from_points for 'quiver' and 'streamlines'.

    Returns arguments (x, y, u, v), to be passed to the underlying matplotlib
    call.

    "u_data" will always be "u_cube.data".
    The matching "v_cube.data" component is stored in kwargs['_v_data'].

    """
    v_data = kwargs.pop("_v_data")

    # Rescale u+v values for plot distortion.
    crs = kwargs.get("transform", None)
    if crs:
        if not isinstance(crs, (ccrs.PlateCarree, ccrs.RotatedPole)):
            msg = (
                "Can only plot vectors provided in a lat-lon "
                'projection, i.e. equivalent to "cartopy.crs.PlateCarree" '
                'or "cartopy.crs.RotatedPole". This '
                "cube coordinate system translates as Cartopy {}."
            )
            raise ValueError(msg.format(crs))
        # Given the above check, the Y points must be latitudes.
        # We therefore **assume** they are in degrees : I'm not sure this
        # is wise, but all the rest of this plot code does that, e.g. in
        # _map_common.
        # TODO: investigate degree units assumptions, here + elsewhere.

        # Implement a latitude scaling, but preserve the given magnitudes.
        u_data, v_data = [arr.copy() for arr in (u_data, v_data)]
        mags = np.sqrt(u_data * u_data + v_data * v_data)
        v_data *= np.cos(np.deg2rad(y_points))
        scales = mags / np.sqrt(u_data * u_data + v_data * v_data)
        u_data *= scales
        v_data *= scales

    return ((x_points, y_points, u_data, v_data), kwargs)


def barbs(u_cube, v_cube, *args, **kwargs):
    """
    Draws a barb plot from two vector component cubes. Triangles, full-lines
    and half-lines represent increments of 50, 10 and 5 respectively.

    Args:

    * u_cube, v_cube : (:class:`~iris.cube.Cube`)
        u and v vector components.  Must have same shape and units.
        If the cubes have geographic coordinates, the values are treated as
        true distance differentials, e.g. windspeeds, and *not* map coordinate
        vectors.  The components are aligned with the North and East of the
        cube coordinate system.

    .. Note::

        At present, if u_cube and v_cube have geographic coordinates, then they
        must be in a lat-lon coordinate system, though it may be a rotated one.
        To transform wind values between coordinate systems, use
        :func:`iris.analysis.cartography.rotate_grid_vectors`.
        To transform coordinate grid points, you will need to create
        2-dimensional arrays of x and y values.  These can be transformed with
        the :meth:`~cartopy.crs.CRS.transform_points` method of
        :class:`cartopy.crs.CRS`.

    Kwargs:

    * coords: (list of :class:`~iris.coords.Coord` or string)
        Coordinates or coordinate names. Use the given coordinates as the axes
        for the plot. The order of the given coordinates indicates which axis
        to use for each, where the first element is the horizontal
        axis of the plot and the second element is the vertical axis
        of the plot.

    * axes: the :class:`matplotlib.axes.Axes` to use for drawing.
        Defaults to the current axes if none provided.

    See :func:`matplotlib.pyplot.barbs` for details of other valid
    keyword arguments.

    """
    #
    # TODO: check u + v cubes for compatibility.
    #
    kwargs["_v_data"] = v_cube.data
    return _draw_2d_from_points(
        "barbs", _vector_component_args, u_cube, *args, **kwargs
    )


def quiver(u_cube, v_cube, *args, **kwargs):
    """
    Draws an arrow plot from two vector component cubes.

    Args:

    * u_cube, v_cube : :class:`~iris.cube.Cube`
        u and v vector components.  Must have same shape and units.
        If the cubes have geographic coordinates, the values are treated as
        true distance differentials, e.g. windspeeds, and *not* map coordinate
        vectors.  The components are aligned with the North and East of the
        cube coordinate system.

    .. Note::

        At present, if u_cube and v_cube have geographic coordinates, then they
        must be in a lat-lon coordinate system, though it may be a rotated one.
        To transform wind values between coordinate systems, use
        :func:`iris.analysis.cartography.rotate_grid_vectors`.
        To transform coordinate grid points, you will need to create
        2-dimensional arrays of x and y values.  These can be transformed with
        the :meth:`~cartopy.crs.CRS.transform_points` method of
        :class:`cartopy.crs.CRS`.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` or string
        Coordinates or coordinate names. Use the given coordinates as the axes
        for the plot. The order of the given coordinates indicates which axis
        to use for each, where the first element is the horizontal
        axis of the plot and the second element is the vertical axis
        of the plot.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    See :func:`matplotlib.pyplot.quiver` for details of other valid
    keyword arguments.

    """
    #
    # TODO: check u + v cubes for compatibility.
    #
    kwargs["_v_data"] = v_cube.data
    return _draw_2d_from_points(
        "quiver", _vector_component_args, u_cube, *args, **kwargs
    )


def plot(*args, **kwargs):
    """
    Draws a line plot based on the given cube(s) or coordinate(s).

    The first one or two arguments may be cubes or coordinates to plot.
    Each of the following is valid::

        # plot a 1d cube against its dimension coordinate
        plot(cube)

        # plot a 1d coordinate
        plot(coord)

        # plot a 1d cube against a given 1d coordinate, with the cube
        # values on the y-axis and the coordinate on the x-axis
        plot(coord, cube)

        # plot a 1d cube against a given 1d coordinate, with the cube
        # values on the x-axis and the coordinate on the y-axis
        plot(cube, coord)

        # plot two 1d coordinates against one-another
        plot(coord1, coord2)

        # plot two 1d cubes against one-another
        plot(cube1, cube2)

    Kwargs:

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    See :func:`matplotlib.pyplot.plot` for details of additional valid
    keyword arguments.

    """
    if "coords" in kwargs:
        raise TypeError(
            '"coords" is not a valid plot keyword. Coordinates '
            "and cubes may be passed as arguments for "
            "full control of the plot axes."
        )
    _plot_args = None
    return _draw_1d_from_points("plot", _plot_args, *args, **kwargs)


def scatter(x, y, *args, **kwargs):
    """
    Draws a scatter plot based on the given cube(s) or coordinate(s).

    Args:

    * x: :class:`~iris.cube.Cube` or :class:`~iris.coords.Coord`
        A cube or a coordinate to plot on the x-axis.

    * y: :class:`~iris.cube.Cube` or :class:`~iris.coords.Coord`
        A cube or a coordinate to plot on the y-axis.

    Kwargs:

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    See :func:`matplotlib.pyplot.scatter` for details of additional
    valid keyword arguments.

    """
    # here we are more specific about argument types than generic 1d plotting
    if not isinstance(x, (iris.cube.Cube, iris.coords.Coord)):
        raise TypeError("x must be a cube or a coordinate.")
    if not isinstance(y, (iris.cube.Cube, iris.coords.Coord)):
        raise TypeError("y must be a cube or a coordinate.")
    args = (x, y) + args
    _plot_args = None
    return _draw_1d_from_points("scatter", _plot_args, *args, **kwargs)


# Provide convenience show method from pyplot
show = plt.show


def symbols(x, y, symbols, size, axes=None, units="inches"):
    """
    Draws fixed-size symbols.

    See :mod:`iris.symbols` for available symbols.

    Args:

    * x: iterable
        The x coordinates where the symbols will be plotted.

    * y: iterable
        The y coordinates where the symbols will be plotted.

    * symbols: iterable
        The symbols (from :mod:`iris.symbols`) to plot.

    * size: float
        The symbol size in `units`.

    Kwargs:

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    * units: ['inches', 'points']
        The unit for the symbol size.

    """
    if axes is None:
        axes = plt.gca()

    offsets = np.array(list(zip(x, y)))

    # XXX "match_original" doesn't work ... so brute-force it instead.
    #   PatchCollection constructor ignores all non-style keywords when using
    #   match_original
    #   See matplotlib.collections.PatchCollection.__init__
    #   Specifically matplotlib/collections line 1053
    # pc = PatchCollection(symbols, offsets=offsets, transOffset=ax.transData,
    #                      match_original=True)
    facecolors = [p.get_facecolor() for p in symbols]
    edgecolors = [p.get_edgecolor() for p in symbols]
    linewidths = [p.get_linewidth() for p in symbols]

    pc = mpl_collections.PatchCollection(
        symbols,
        offsets=offsets,
        transOffset=axes.transData,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )

    if units == "inches":
        scale = axes.figure.dpi
    elif units == "points":
        scale = axes.figure.dpi / 72.0
    else:
        raise ValueError("Unrecognised units: '%s'" % units)
    pc.set_transform(mpl_transforms.Affine2D().scale(0.5 * size * scale))

    axes.add_collection(pc)
    axes.autoscale_view()


def citation(text, figure=None, axes=None):
    """
    Add a text citation to a plot.

    Places an anchored text citation in the bottom right
    hand corner of the plot.

    Args:

    * text: str
        Citation text to be plotted.

    Kwargs:

    * figure::class:`matplotlib.figure.Figure`
        Target figure instance. Defaults to the current figure if none provided.

    * axes: :class:`matplotlib.axes.Axes`
        The axes to use for drawing.  Defaults to the current axes if none
        provided.

    """

    if text is not None and len(text):
        if figure is None and not axes:
            figure = plt.gcf()
        anchor = AnchoredText(text, prop=dict(size=6), frameon=True, loc=4)
        anchor.patch.set_boxstyle("round, pad=0, rounding_size=0.2")
        axes = axes if axes else figure.gca()
        axes.add_artist(anchor)
