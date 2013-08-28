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
Iris-specific extensions to matplotlib, mimicking the :mod:`matplotlib.pyplot`
interface.

See also: :ref:`matplotlib <matplotlib:users-guide-index>`.

"""

import collections
import datetime
import functools
import warnings

import matplotlib.axes
import matplotlib.collections as mpl_collections
import matplotlib.dates as mpl_dates
import matplotlib.transforms as mpl_transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import numpy as np
import numpy.ma as ma
import cartopy.crs
import cartopy.mpl.geoaxes


import iris.cube
import iris.coord_systems
import iris.analysis.cartography as cartography
import iris.coords
import iris.palette
import iris.unit


# Cynthia Brewer citation text.
BREWER_CITE = 'Colours based on ColorBrewer.org'


PlotDefn = collections.namedtuple('PlotDefn', ('coords', 'transpose'))


def _get_plot_defn_custom_coords_picked(cube, coords, mode, ndims=2):
    def as_coord(coord):
        if isinstance(coord, basestring):
            coord = cube.coord(name=coord)
        else:
            coord = cube.coord(coord=coord)
        return coord
    coords = map(as_coord, coords)

    # Check that we were given the right number of coordinates
    if len(coords) != ndims:
        coord_names = ', '.join([coord.name() for coord in coords])
        raise ValueError('The list of coordinates given (%s) should have the'
                         ' same length (%s) as the dimensionality of the'
                         ' required plot (%s)' % (coord_names,
                                                  len(coords), ndims))

    # Check which dimensions are spanned by each coordinate.
    get_span = lambda coord: set(cube.coord_dims(coord))
    spans = map(get_span, coords)
    for span, coord in zip(spans, coords):
        if not span:
            msg = 'The coordinate {!r} doesn\'t span a data dimension.'
            raise ValueError(msg.format(coord.name()))
        if mode == iris.coords.BOUND_MODE and len(span) != 1:
            raise ValueError('The coordinate {!r} is multi-dimensional and'
                             ' cannot be used in a cell-based plot.'
                             .format(coord.name()))

    # Check the combination of coordinates spans enough (ndims) data
    # dimensions.
    total_span = set().union(*spans)
    if len(total_span) != ndims:
        coord_names = ', '.join([coord.name() for coord in coords])
        raise ValueError('The given coordinates ({}) don\'t span the {} data'
                         ' dimensions.'.format(coord_names, ndims))

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


def _valid_bound_coord(coord):
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
        msg = 'Cube must be %s-dimensional. Got %s dimensions.'
        raise ValueError(msg % (ndims, cube.ndim))

    # Start by taking the DimCoords from each dimension.
    coords = [None] * ndims
    for dim_coord in cube.dim_coords:
        dim = cube.coord_dims(dim_coord)[0]
        coords[dim] = dim_coord

    # When appropriate, restrict to 1D with bounds.
    if mode == iris.coords.BOUND_MODE:
        coords = map(_valid_bound_coord, coords)

    def guess_axis(coord):
        axis = None
        if coord is not None:
            axis = iris.util.guess_coord_axis(coord)
        return axis

    # Allow DimCoords in aux_coords to fill in for missing dim_coords.
    for dim, coord in enumerate(coords):
        if coord is None:
            aux_coords = cube.coords(dimensions=dim)
            aux_coords = filter(lambda coord:
                                isinstance(coord, iris.coords.DimCoord),
                                aux_coords)
            if aux_coords:
                key_func = lambda coord: coord._as_defn()
                aux_coords.sort(key=key_func)
                coords[dim] = aux_coords[0]

    if mode == iris.coords.POINT_MODE:
        # Allow multi-dimensional aux_coords to override the dim_coords
        # along the Z axis. This results in a preference for using the
        # derived altitude over model_level_number or level_height.
        # Limit to Z axis to avoid preferring latitude over grid_latitude etc.
        axes = map(guess_axis, coords)
        axis = 'Z'
        if axis in axes:
            for coord in cube.coords(dim_coords=False):
                if max(coord.shape) > 1 and \
                        iris.util.guess_coord_axis(coord) == axis:
                    coords[axes.index(axis)] = coord

    # Re-order the coordinates to achieve the preferred
    # horizontal/vertical associations.
    def sort_key(coord):
        order = {'X': 2, 'T': 1, 'Y': -1, 'Z': -2}
        axis = guess_axis(coord)
        return (order.get(axis, 0), coord and coord.name())
    sorted_coords = sorted(coords, key=sort_key)

    transpose = (sorted_coords != coords)
    return PlotDefn(sorted_coords, transpose)


def _can_draw_map(plot_coords):
    std_names = [coord and coord.standard_name for coord in plot_coords]
    valid_std_names = [
        ['latitude', 'longitude'],
        ['grid_latitude', 'grid_longitude'],
        ['projection_y_coordinate', 'projection_x_coordinate']
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


def _draw_2d_from_bounds(draw_method_name, cube, *args, **kwargs):
    # NB. In the interests of clarity we use "u" and "v" to refer to the
    # horizontal and vertical axes on the matplotlib plot.
    mode = iris.coords.BOUND_MODE
    # Get & remove the coords entry from kwargs.
    coords = kwargs.pop('coords', None)
    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode)
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)

    if _can_draw_map(plot_defn.coords):
        result = _map_common(draw_method_name, None, iris.coords.BOUND_MODE,
                             cube, plot_defn, *args, **kwargs)
    else:
        # Obtain data array.
        data = cube.data
        if plot_defn.transpose:
            data = data.T

        # Obtain U and V coordinates
        v_coord, u_coord = plot_defn.coords

        # XXX: Watch out for non-contiguous bounds.
        if u_coord:
            u = u_coord.contiguous_bounds()
        else:
            u = np.arange(data.shape[1] + 1)
        if v_coord:
            v = v_coord.contiguous_bounds()
        else:
            v = np.arange(data.shape[0] + 1)

        if plot_defn.transpose:
            u = u.T
            v = v.T

        u, v = _broadcast_2d(u, v)
        draw_method = getattr(plt, draw_method_name)
        result = draw_method(u, v, data, *args, **kwargs)

    return result


def _draw_2d_from_points(draw_method_name, arg_func, cube, *args, **kwargs):
    # NB. In the interests of clarity we use "u" and "v" to refer to the
    # horizontal and vertical axes on the matplotlib plot.
    mode = iris.coords.POINT_MODE
    # Get & remove the coords entry from kwargs.
    coords = kwargs.pop('coords', None)
    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode)
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)

    if _can_draw_map(plot_defn.coords):
        result = _map_common(draw_method_name, arg_func,
                             iris.coords.POINT_MODE, cube, plot_defn,
                             *args, **kwargs)
    else:
        # Obtain data array.
        data = cube.data
        if plot_defn.transpose:
            data = data.T

        # Obtain U and V coordinates
        v_coord, u_coord = plot_defn.coords
        if u_coord:
            u = u_coord.points
            u = _fixup_dates(u_coord, u)
        else:
            u = np.arange(data.shape[1])
        if v_coord:
            v = v_coord.points
            v = _fixup_dates(v_coord, v)
        else:
            v = np.arange(data.shape[0])

        if plot_defn.transpose:
            u = u.T
            v = v.T

        if u.dtype == np.dtype(object) and isinstance(u[0], datetime.datetime):
            u = mpl_dates.date2num(u)

        if v.dtype == np.dtype(object) and isinstance(v[0], datetime.datetime):
            v = mpl_dates.date2num(v)

        u, v = _broadcast_2d(u, v)

        draw_method = getattr(plt, draw_method_name)
        if arg_func is not None:
            args, kwargs = arg_func(u, v, data, *args, **kwargs)
            result = draw_method(*args, **kwargs)
        else:
            result = draw_method(u, v, data, *args, **kwargs)

    return result


def _fixup_dates(coord, values):
    if coord.units.calendar is not None and values.ndim == 1:
        r = [datetime.datetime(*(coord.units.num2date(val).timetuple()[0:6]))
             for val in values]
        values = np.empty(len(r), dtype=object)
        values[:] = r
    return values


def _data_from_coord_or_cube(c):
    if isinstance(c, iris.cube.Cube):
        data = c.data
    elif isinstance(c, iris.coords.Coord):
        data = _fixup_dates(c, c.points)
    else:
        raise TypeError('Plot arguments must be cubes or coordinates.')
    return data


def _uv_from_u_object_v_object(u_object, v_object):
    ndim_msg = 'Cube or coordinate must be 1-dimensional. Got {} dimensions.'
    if u_object is not None and u_object.ndim > 1:
        raise ValueError(ndim_msg.format(u_object.ndim))
    if v_object.ndim > 1:
        raise ValueError(ndim_msg.format(v_object.ndim))
    type_msg = 'Plot arguments must be cubes or coordinates.'
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
        u_object, = plot_defn.coords
    return u_object


def _get_plot_objects(args):
    if len(args) > 1 and isinstance(args[1],
                                    (iris.cube.Cube, iris.coords.Coord)):
        # two arguments
        u_object, v_object = args[:2]
        u, v = _uv_from_u_object_v_object(*args[:2])
        args = args[2:]
        if len(u) != len(v):
            msg = "The x and y-axis objects are not compatible. They should " \
                  "have equal sizes but got ({}: {}) and ({}: {})."
            raise ValueError(msg.format(u_object.name(), len(u),
                                        v_object.name(), len(v)))
    else:
        # single argument
        v_object = args[0]
        u_object = _u_object_from_v_object(v_object)
        u, v = _uv_from_u_object_v_object(u_object, args[0])
        args = args[1:]
    return u_object, v_object, u, v, args


def _draw_1d_from_points(draw_method_name, arg_func, *args, **kwargs):
    # NB. In the interests of clarity we use "u" to refer to the horizontal
    # axes on the matplotlib plot and "v" for the vertical axes.

    # retrieve the objects that are plotted on the horizontal and vertical
    # axes (cubes or coordinates) and their respective values, along with the
    # argument tuple with these objects removed
    u_object, v_object, u, v, args = _get_plot_objects(args)

    # if both u_object and v_object are coordinates then check if a map
    # should be drawn
    if isinstance(u_object, iris.coords.Coord) and \
            isinstance(v_object, iris.coords.Coord) and \
            _can_draw_map([v_object, u_object]):
        # Replace non-cartopy subplot/axes with a cartopy alternative and set
        # the transform keyword.
        draw_method, kwargs = _geoaxes_draw_method_and_kwargs(u_object,
                                                              v_object,
                                                              draw_method_name,
                                                              kwargs)
    else:
        # just use a pyplot function to draw
        draw_method = getattr(plt, draw_method_name)

    if arg_func is not None:
        args, kwargs = arg_func(u, v, *args, **kwargs)
        result = draw_method(*args, **kwargs)
    else:
        result = draw_method(u, v, *args, **kwargs)

    return result


def _get_cartopy_axes(cartopy_proj):
    # Replace non-cartopy subplot/axes with a cartopy alternative.
    ax = plt.gca()
    if not isinstance(ax,
                      cartopy.mpl.geoaxes.GeoAxes):
        fig = plt.gcf()
        if isinstance(ax, matplotlib.axes.SubplotBase):
            new_ax = fig.add_subplot(ax.get_subplotspec(),
                                     projection=cartopy_proj,
                                     title=ax.get_title(),
                                     xlabel=ax.get_xlabel(),
                                     ylabel=ax.get_ylabel())
        else:
            new_ax = fig.add_axes(projection=cartopy_proj,
                                  title=ax.get_title(),
                                  xlabel=ax.get_xlabel(),
                                  ylabel=ax.get_ylabel())

        # delete the axes which didn't have a cartopy projection
        fig.delaxes(ax)
        ax = new_ax
    return ax


def _geoaxes_draw_method_and_kwargs(x_coord, y_coord, draw_method_name,
                                    kwargs):
    """
    Retrieve a GeoAxes draw method and appropriate keyword arguments for
    calling it given the coordinates and existing keywords.

    """
    if x_coord.coord_system != y_coord.coord_system:
        raise ValueError('The X and Y coordinates must have equal coordinate'
                         ' systems.')
    cs = x_coord.coord_system
    if cs is not None:
        cartopy_proj = cs.as_cartopy_projection()
    else:
        cartopy_proj = cartopy.crs.PlateCarree()
    ax = _get_cartopy_axes(cartopy_proj)
    draw_method = getattr(ax, draw_method_name)
    # Set the "from transform" keyword.
    new_kwargs = kwargs.copy()
    assert 'transform' not in new_kwargs, 'Transform keyword is not allowed.'
    new_kwargs['transform'] = cartopy_proj

    return draw_method, new_kwargs


def _map_common(draw_method_name, arg_func, mode, cube, plot_defn,
                *args, **kwargs):
    """
    Draw the given cube on a map using its points or bounds.

    "Mode" parameter will switch functionality between POINT or BOUND plotting.


    """
    # Generate 2d x and 2d y grids.
    y_coord, x_coord = plot_defn.coords
    if mode == iris.coords.POINT_MODE:
        if x_coord.ndim == y_coord.ndim == 1:
            x, y = np.meshgrid(x_coord.points, y_coord.points)
        elif x_coord.ndim == y_coord.ndim == 2:
            x = x_coord.points
            y = y_coord.points
        else:
            raise ValueError("Expected 1D or 2D XY coords")
    else:
        try:
            x, y = np.meshgrid(x_coord.contiguous_bounds(),
                               y_coord.contiguous_bounds())
        # Exception translation.
        except iris.exceptions.CoordinateMultiDimError:
            raise ValueError("Could not get XY grid from bounds. "
                             "X or Y coordinate not 1D.")
        except ValueError:
            raise ValueError("Could not get XY grid from bounds. "
                             "X or Y coordinate doesn't have 2 bounds "
                             "per point.")

    # Obtain the data array.
    data = cube.data
    if plot_defn.transpose:
        data = data.T

    # If we are global, then append the first column of data the array to the
    # last (and add 360 degrees) NOTE: if it is found that this block of code
    # is useful in anywhere other than this plotting routine, it may be better
    # placed in the CS.
    if getattr(x_coord, 'circular', False):
        _, direction = iris.util.monotonic(x_coord.points,
                                           return_direction=True)
        y = np.append(y, y[:, 0:1], axis=1)
        x = np.append(x, x[:, 0:1] + 360 * direction, axis=1)
        data = ma.concatenate([data, data[:, 0:1]], axis=1)

    # Replace non-cartopy subplot/axes with a cartopy alternative and set the
    # transform keyword.
    draw_method, kwargs = _geoaxes_draw_method_and_kwargs(x_coord, y_coord,
                                                          draw_method_name,
                                                          kwargs)

    if arg_func is not None:
        new_args, kwargs = arg_func(x, y, data, *args, **kwargs)
    else:
        new_args = (x, y, data) + args

    # Draw the contour lines/filled contours.
    return draw_method(*new_args, **kwargs)


def contour(cube, *args, **kwargs):
    """
    Draws contour lines based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    See :func:`matplotlib.pyplot.contour` for details of other valid keyword
    arguments.

    """
    result = _draw_2d_from_points('contour', None, cube, *args, **kwargs)
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

    See :func:`matplotlib.pyplot.contourf` for details of other valid keyword
    arguments.

    """
    coords = kwargs.get('coords')
    kwargs.setdefault('antialiased', True)
    result = _draw_2d_from_points('contourf', None, cube, *args, **kwargs)

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
        if result.extend == 'neither':
            levels = levels[1:-1]
            colors = colors[:-1]
        elif result.extend == 'min':
            levels = levels[:-1]
            colors = colors[:-1]
        elif result.extend == 'max':
            levels = levels[1:]
            colors = colors[:-1]
        else:
            colors = colors[:-1]
        if len(levels) > 0:
            # Draw the lines just *below* the polygons to ensure we minimise
            # any boundary shift.
            zorder = result.collections[0].zorder - 1
            contour(cube, levels=levels, colors=colors, antialiased=True,
                    zorder=zorder, coords=coords)

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

     * mode - Either ``iris.coords.POINT_MODE`` or ``iris.coords.BOUND_MODE``.
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
    orography = cube.coord('surface_altitude')

    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode,
                                                        ndims=2)
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)
    v_coord, u_coord = plot_defn.coords

    # Find which plot coordinate corresponds to the derived altitude, so that
    # we can replace altitude with the surface altitude.
    if v_coord and v_coord.standard_name == 'altitude':
        # v is altitude, so plot u and orography with orog in the y direction.
        result = vert_plot(u_coord, orography, style_args)
    elif u_coord and u_coord.standard_name == 'altitude':
        # u is altitude, so plot v and orography with orog in the x direction.
        result = horiz_plot(v_coord, orography, style_args)
    else:
        raise ValueError('Plot does not use hybrid height. One of the '
                         'coordinates to plot must be altitude, but %s and %s '
                         'were given.' % (u_coord.name(), v_coord.name()))
    return result


def orography_at_bounds(cube, facecolor='#888888', coords=None):
    """Plots orography defined at cell boundaries from the given Cube."""

    # XXX Needs contiguous orography corners to work.
    raise NotImplementedError('This operation is temporarily not provided '
                              'until coordinates can expose 2d contiguous '
                              'bounds (corners).')

    style_args = {'edgecolor': 'none', 'facecolor': facecolor}

    def vert_plot(u_coord, orography, style_args):
        u = u_coord.contiguous_bounds()
        left = u[:-1]
        height = orography.points
        width = u[1:] - left
        return plt.bar(left, height, width, **style_args)

    def horiz_plot(v_coord, orography, style_args):
        v = v_coord.contiguous_bounds()
        bottom = v[:-1]
        width = orography.points
        height = v[1:] - bottom
        return plt.barh(bottom, width, height, **style_args)

    return _fill_orography(cube, coords, iris.coords.BOUND_MODE, vert_plot,
                           horiz_plot, style_args)


def orography_at_points(cube, facecolor='#888888', coords=None):
    """Plots orography defined at sample points from the given Cube."""

    style_args = {'facecolor': facecolor}

    def vert_plot(u_coord, orography, style_args):
        x = u_coord.points
        y = orography.points
        return plt.fill_between(x, y, **style_args)

    def horiz_plot(v_coord, orography, style_args):
        y = v_coord.points
        x = orography.points
        return plt.fill_betweenx(y, x, **style_args)

    return _fill_orography(cube, coords, iris.coords.POINT_MODE, vert_plot,
                           horiz_plot, style_args)


def outline(cube, coords=None):
    """
    Draws cell outlines based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    """
    result = _draw_2d_from_bounds('pcolormesh', cube, facecolors='none',
                                  edgecolors='k', antialiased=True,
                                  coords=coords)
    # set the _is_stroked property to get a single color grid.
    # See https://github.com/matplotlib/matplotlib/issues/1302
    result._is_stroked = False
    if hasattr(result, '_wrapped_collection_fix'):
        result._wrapped_collection_fix._is_stroked = False
    return result


def pcolor(cube, *args, **kwargs):
    """
    Draws a pseudocolor plot based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    See :func:`matplotlib.pyplot.pcolor` for details of other valid keyword
    arguments.

    """
    kwargs.setdefault('antialiased', True)
    result = _draw_2d_from_bounds('pcolor', cube, *args, **kwargs)
    return result


def pcolormesh(cube, *args, **kwargs):
    """
    Draws a pseudocolor plot based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    See :func:`matplotlib.pyplot.pcolormesh` for details of other valid keyword
    arguments.

    """
    result = _draw_2d_from_bounds('pcolormesh', cube, *args, **kwargs)
    return result


def points(cube, *args, **kwargs):
    """
    Draws sample point positions based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    See :func:`matplotlib.pyplot.scatter` for details of other valid keyword
    arguments.

    """
    _scatter_args = lambda u, v, data, *args, **kwargs: ((u, v) + args, kwargs)
    return _draw_2d_from_points('scatter', _scatter_args, cube,
                                *args, **kwargs)


def _1d_coords_deprecation_handler(func):
    """
    Manage the deprecation of the coords keyword argument to 1d plot
    functions.

    """
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        coords = kwargs.pop('coords', None)
        if coords is not None:
            # issue a deprecation warning and check to see if the old
            # interface should be mimicked for the deprecation period
            warnings.warn('The coords keyword argument is deprecated.',
                          stacklevel=2)
            if len(coords) != 1:
                msg = 'The list of coordinates given should have length 1 ' \
                      'but it has length {}.'
                raise ValueError(msg.format(len(coords)))
            if isinstance(args[0], iris.cube.Cube):
                if len(args) < 2 or not isinstance(args[1], (iris.cube.Cube,
                                                   iris.coords.Coord)):
                    if isinstance(coords[0], basestring):
                        coord = args[0].coord(name=coords[0])
                    else:
                        coord = args[0].coord(coord=coords[0])
                    if not args[0].coord_dims(coord):
                        raise ValueError("The coordinate {!r} doesn't "
                                         "span a data dimension."
                                         "".format(coord.name()))
                    args = (coord,) + args
        return func(*args, **kwargs)
    return _wrapper


@_1d_coords_deprecation_handler
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

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

        .. deprecated:: 1.5

           The plot coordinates can be specified explicitly as in the
           above examples, so this keyword is no longer needed.

    See :func:`matplotlib.pyplot.plot` for details of valid keyword
    arguments.

    """
    _plot_args = None
    return _draw_1d_from_points('plot', _plot_args, *args, **kwargs)


def scatter(x, y, *args, **kwargs):
    """
    Draws a scatter plot based on the given cube(s) or coordinate(s).

    Args:

    * x: :class:`~iris.cube.Cube` or :class:`~iris.coords.Coord`
        A cube or a coordinate to plot on the x-axis.

    * y: :class:`~iris.cube.Cube` or :class:`~iris.coords.Coord`
        A cube or a coordinate to plot on the y-axis.

    See :func:`matplotlib.pyplot.scatter` for details of valid keyword
    arguments.

    """
    # here we are more specific about argument types than generic 1d plotting
    if not isinstance(x, (iris.cube.Cube, iris.coords.Coord)):
        raise TypeError('x must be a cube or a coordinate.')
    if not isinstance(y, (iris.cube.Cube, iris.coords.Coord)):
        raise TypeError('y must be a cube or a coordinate.')
    args = (x, y) + args
    _plot_args = None
    return _draw_1d_from_points('scatter', _plot_args, *args, **kwargs)


# Provide convenience show method from pyplot
show = plt.show


def symbols(x, y, symbols, size, axes=None, units='inches'):
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

    * axes:
        The :class:`matplotlib.axes.Axes` in which the symbols will be added.
        Defaults to the current axes.

    * units: ['inches', 'points']
        The unit for the symbol size.

    """
    if axes is None:
        axes = plt.gca()

    offsets = np.array(zip(x, y))

    # XXX "match_original" doesn't work ... so brute-force it instead.
    #   PatchCollection constructor ignores all non-style keywords when using
    #   match_original
    #   See matplotlib.collections.PatchCollection.__init__
    #   Specifically matplotlib/collections line 1053
    #pc = PatchCollection(symbols, offsets=offsets, transOffset=ax.transData,
    #                     match_original=True)
    facecolors = [p.get_facecolor() for p in symbols]
    edgecolors = [p.get_edgecolor() for p in symbols]
    linewidths = [p.get_linewidth() for p in symbols]

    pc = mpl_collections.PatchCollection(symbols, offsets=offsets,
                                         transOffset=axes.transData,
                                         facecolors=facecolors,
                                         edgecolors=edgecolors,
                                         linewidths=linewidths)

    if units == 'inches':
        scale = axes.figure.dpi
    elif units == 'points':
        scale = axes.figure.dpi / 72.0
    else:
        raise ValueError("Unrecognised units: '%s'" % units)
    pc.set_transform(mpl_transforms.Affine2D().scale(0.5 * size * scale))

    axes.add_collection(pc)
    axes.autoscale_view()


def citation(text, figure=None):
    """
    Add a text citation to a plot.

    Places an anchored text citation in the bottom right
    hand corner of the plot.

    Args:

    * text:
        Citation text to be plotted.

    Kwargs:

    * figure:
        Target :class:`matplotlib.figure.Figure` instance. Defaults
        to the current figure if none provided.

    """

    if text is not None and len(text):
        if figure is None:
            figure = plt.gcf()
        anchor = AnchoredText(text, prop=dict(size=6), frameon=True, loc=4)
        anchor.patch.set_boxstyle('round, pad=0, rounding_size=0.2')
        figure.gca().add_artist(anchor)
