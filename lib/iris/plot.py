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
Iris-specific extensions to matplotlib, mimicking the :mod:`matplotlib.pyplot`
interface.

See also: :ref:`matplotlib <matplotlib:users-guide-index>`, :ref:`cartopy <cartopy:users-guide-index>`.

"""

import collections
import datetime
import warnings

import matplotlib.axes
import matplotlib.collections as mpl_collections
import matplotlib.dates as mpl_dates
import matplotlib.transforms as mpl_transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import numpy
import numpy.ma
import cartopy.crs
import cartopy.mpl_integration.geoaxes


import iris.cube
import iris.coord_systems
import iris.analysis.cartography
import iris.coords
import iris.palette
import iris.unit


# Cynthia Brewer citation text.
_BREWER = 'Colours based on ColorBrewer.org'


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
        raise ValueError(msg % (ndims, cube.data.ndim))

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
            aux_coords = filter(lambda coord: isinstance(coord, iris.coords.DimCoord), aux_coords)
            if aux_coords:
                key_func = lambda coord: coord._as_defn()
                aux_coords.sort(key=key_func)
                coords[dim] = aux_coords[0]

    if mode == iris.coords.POINT_MODE:
        # Allow multi-dimensional aux_coords to override the dim_coords. (things like
        # grid_latitude will be overriden by latitude etc.)
        axes = map(guess_axis, coords)
        for coord in cube.coords(dim_coords=False):
            if max(coord.shape) > 1 and (mode == iris.coords.POINT_MODE or
                                         coord.nbounds):
                axis = iris.util.guess_coord_axis(coord)
                if axis and axis in axes:
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
    u = numpy.atleast_2d(u)
    v = numpy.atleast_2d(v.T).T
    u, v = numpy.broadcast_arrays(u, v)
    return u, v


def _draw_2d_from_bounds(draw_method_name, cube, *args, **kwargs):
    # NB. In the interests of clarity we use "u" and "v" to refer to the horizontal and vertical
    # axes on the matplotlib plot.
    
    # get & remove the coords entry from kwargs
    coords = kwargs.pop('coords', None)
    mode = iris.coords.BOUND_MODE
    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode)
        if plot_defn.coords[0].nbounds == 0 or plot_defn.coords[1].nbounds == 0:
            raise ValueError('Cannot draw %r against %r using %r as at least'
                             ' one of them is not bounded.' % (
                                plot_defn.coords[0].name(),
                                plot_defn.coords[1].name(),
                                draw_method_name))
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)
    
    data = cube.data
    if plot_defn.transpose:
        data = data.T

    if _can_draw_map(plot_defn.coords):
        result = _map_common(draw_method_name, None, iris.coords.BOUND_MODE, cube, data, *args, **kwargs)
    else:
        # Obtain U and V coordinates
        v_coord, u_coord = plot_defn.coords

        # XXX: Watch out for non-contiguous bounds.
        if u_coord:
            u = u_coord.contiguous_bounds()
        else:
            u = numpy.arange(data.shape[1] + 1)
        if v_coord:
            v = v_coord.contiguous_bounds()
        else:
            v = numpy.arange(data.shape[0] + 1)

        if plot_defn.transpose:
            u = u.T
            v = v.T

        u, v = _broadcast_2d(u, v)

        draw_method = getattr(plt, draw_method_name)
        result = draw_method(u, v, data, *args, **kwargs)

    return result


def _draw_2d_from_points(draw_method_name, arg_func, cube, *args, **kwargs):
    # NB. In the interests of clarity we use "u" and "v" to refer to the horizontal and vertical
    # axes on the matplotlib plot.

    # get & remove the coords entry from kwargs
    coords = kwargs.pop('coords', None)
    mode = iris.coords.POINT_MODE
    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode)
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)
        
    data = cube.data
    if plot_defn.transpose:
        data = data.T

    if _can_draw_map(plot_defn.coords):
        result = _map_common(draw_method_name, arg_func, iris.coords.POINT_MODE, cube, data, *args, **kwargs)
    else:
        # Obtain U and V coordinates
        v_coord, u_coord = plot_defn.coords
        if u_coord:
            u = u_coord.points
            u = _fixup_dates(u_coord, u)
        else:
            u = numpy.arange(data.shape[1])
        if v_coord:
            v = v_coord.points
            v = _fixup_dates(v_coord, v)
        else:
            v = numpy.arange(data.shape[0])

        if plot_defn.transpose:
            u = u.T
            v = v.T

        if u.dtype == numpy.dtype(object) and isinstance(u[0], datetime.datetime):
            u = mpl_dates.date2num(u)
        
        if v.dtype == numpy.dtype(object) and isinstance(v[0], datetime.datetime):
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
        if coord.units.calendar not in [iris.unit.CALENDAR_GREGORIAN, iris.unit.CALENDAR_STANDARD]:
            # TODO #435 Requires fix so that matplotlib doesn't dismiss the calendar
            warnings.warn('Calendar info dismissed when passing to Matplotlib.')
        r = [datetime.datetime(*(coord.units.num2date(val).timetuple()[0:6])) for val in values]
        values = numpy.empty(len(r), dtype=object)
        values[:] = r
    return values


def _draw_1d_from_points(draw_method_name, arg_func, cube, *args, **kwargs):
    # NB. In the interests of clarity we use "u" to refer to the horizontal
    # axes on the matplotlib plot.

    # get & remove the coords entry from kwargs
    coords = kwargs.pop('coords', None)
    mode = iris.coords.POINT_MODE
    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode, ndims=1)
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=1)

    data = cube.data

    # Obtain U coordinates
    u_coord, = plot_defn.coords
    if u_coord:
        u = u_coord.points
        u = _fixup_dates(u_coord, u)
    else:
        u = numpy.arange(data.shape[0])

    draw_method = getattr(plt, draw_method_name)
    if arg_func is not None:
        args, kwargs = arg_func(u, data, *args, **kwargs)
        result = draw_method(*args, **kwargs)
    else:
        result = draw_method(u, data, *args, **kwargs)

    return result


def _get_cartopy_axes(cartopy_proj):
    # Replace non-cartopy subplot/axes with a cartopy alternative.
    # XXX original subplot properties will be lost...
    # XXX consider allowing the axes to be passed through
    ax = plt.gca()
    if not isinstance(ax,
                      cartopy.mpl_integration.geoaxes.GenericProjectionAxes):
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


def _map_common(draw_method_name, arg_func, mode, cube, data, *args, **kwargs):
    """
    Draw the given cube on a map using its points or bounds.
    
    "Mode" parameter will switch functionality between POINT or BOUND plotting.
    
    """
    # get the 2d x and 2d y from the CS
    if mode == iris.coords.POINT_MODE:
        x, y = iris.analysis.cartography.get_xy_grids(cube)
    else:
        x, y = iris.analysis.cartography.get_xy_contiguous_bounded_grids(cube)

    # take a copy of the data so that we can make modifications to it
    data = data.copy()

    # If we are global, then append the first column of data the array to the last (and add 360 degrees)  	  	 
    # NOTE: if it is found that this block of code is useful in anywhere other than this plotting routine, it  	  	 
    # may be better placed in the CS.
    x_coord = cube.coord(axis="X")
    if getattr(x_coord, 'circular', False):
        y = numpy.append(y, y[:, 0:1], axis=1)
        x = numpy.append(x, x[:, 0:1] + 360, axis=1)
        data = numpy.ma.concatenate([data, data[:, 0:1]], axis=1)

    # Get the native crs and map (might be the same cartopy definiton)
    cs = cube.coord_system('CoordSystem')
    cartopy_crs = cs.as_cartopy_crs()  # E.g. Geodetic
    cartopy_proj = cs.as_cartopy_projection()  # E.g. PlateCarree

    # Replace non-cartopy subplot/axes with a cartopy alternative.
    ax = _get_cartopy_axes(cartopy_proj)

    draw_method = getattr(ax, draw_method_name)

    # Set the "from transform" keyword. 
    assert 'transform' not in kwargs, 'Transform keyword is not allowed.' 
    kwargs['transform'] = cartopy_crs
    
    if arg_func is not None:
        new_args, kwargs = arg_func(x, y, data, *args, **kwargs)
    else:
        new_args = (x, y, data) + args

    # Draw the contour lines/filled contours.
    return draw_method(*new_args, **kwargs)


@iris.palette.auto_palette
def contour(cube, *args, **kwargs):
    """
    Draws contour lines based on the given Cube.
    
    Args:
    
    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the given coordinates
        indicates which axis to use for each, where the first element is the horizontal axis of
        the plot and the second element is the vertical axis of the plot.
        
    See :func:`matplotlib.pyplot.contour` for details of other valid keyword arguments.
    
    """
    result =_draw_2d_from_points('contour', None, cube, *args, **kwargs)
    if iris.palette.is_brewer(kwargs.get('cmap')):
        citation(_BREWER)
    return result


@iris.palette.auto_palette
def contourf(cube, *args, **kwargs):
    """
    Draws filled contours based on the given Cube.
    
    Args:
    
    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the given coordinates
        indicates which axis to use for each, where the first element is the horizontal axis of
        the plot and the second element is the vertical axis of the plot.
        
    See :func:`matplotlib.pyplot.contourf` for details of other valid keyword arguments.
    
    """
    coords = kwargs.get('coords')
    kwargs.setdefault('antialiased', True)
    result = _draw_2d_from_points('contourf', None, cube, *args, **kwargs)

    if iris.palette.is_brewer(kwargs.get('cmap')):
        citation(_BREWER)
    
    # Matplotlib produces visible seams between anti-aliased polygons.
    # But if the polygons are virtually opaque then we can cover the seams
    # by drawing anti-aliased lines *underneath* the polygon joins.

    # Figure out the alpha level for the contour plot
    if result.alpha is None:
        alpha = result.collections[0].get_facecolor()[0][3]
    else:
        alpha = result.alpha
    # If the contours are anti-aliased and mostly opaque then draw lines under the seams.
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
            # Draw the lines just *below* the polygons to ensure we minimise any boundary shift.
            zorder = result.collections[0].zorder - 1
            contour(cube, levels=levels, colors=colors, antialiased=True, zorder=zorder, coords=coords)

    return result


def map_setup(projection=None, xlim=None, ylim=None, cube=None, mode=None):
    """
    Setup matplotlib for cartographic plotting.
    
    The projection is taken from the projection keyword, if present,
    otherwise it is taken from the cube, if present,
    otherwise it defaults to a PlateCarree projection.
    
    The xy limits are taken from the xlim and ylim keywords, if present,
    otherwise they are taken from the cube, if present,
    otherwise it is left for matplotlib to set limit during the first plot.
    
    Kwargs:
    
        * projection - The projection to use for plotting.
        * xlim       - x limits
        * ylim       - y limits
        * cube       - A cube which can be used for projection and limits.
        * mode       - Controls min/max calulation for bounded coordinates
                       (Passed through to :func:`~iris.analysis.cartography.xy_range`). 
                       Set to iris.coords.POINT_MODE or iris.coords.BOUND_MODE.
                       Default is iris.coords.BOUND_MODE.

    Note:

        There is no need to setup a map in many cases,
        as this is done for you by Iris' plotting routines.

            # Unnecessary map_setup
            map_setup(cube)
            contourf(cube)
    
        Suggested uses of this function are as follows, ::

            # Alternative projection, automatic limits
            map_setup(projection)
            contourf(cube)
        
            # Alternative projection, custom limits
            map_setup(projection, xlim, ylim)
            contourf(cube)
            
            # Native projection, custom limits
            map_setup(cube, xlim, ylim)
            contourf(cube)
            
        The following snippet is inefficient.
        Coordinate transformation is used twice, which is not necessary. ::
    
            # Transforms to calulate projected extents
            map_setup(cube, other_projection)
            # Transforms again, for plotting
            contourf(cube)
            
        Instead, let the limits be calulated automatically,
        as in the suggested usage, above.

    """  
    # Which projection?
    if projection is None and cube is not None:
        cs = cube.coord_system("CoordSystem")
        projection = cs.as_cartopy_projection() if cs else None
    if projection is None:
        projection = cartopy.crs.PlateCarree()
    
    lim_crs = projection
    
    # Which extents?
    if (xlim is None or ylim is None) and cube is not None:
        mode = mode or iris.coords.BOUND_MODE
        extents = iris.analysis.cartography.xy_range(cube, mode, projection)
        xlim = extents[0]
        ylim = extents[1]
        lim_crs = cs.as_cartopy_crs() if cs else None

    ax = _get_cartopy_axes(projection)

    if xlim is not None != ylim is not None:
        warnings.warn('Both xlim and ylim must currently be set.')
    
    if xlim is not None:
        ax.set_extent(tuple(xlim) + tuple(ylim), lim_crs)
    
    return ax


def _fill_orography(cube, coords, mode, vert_plot, horiz_plot, style_args):
    # Find the orography coordinate.
    orography = cube.coord('surface_altitude')

    if coords is not None:
        plot_defn = _get_plot_defn_custom_coords_picked(cube, coords, mode, ndims=2)
    else:
        plot_defn = _get_plot_defn(cube, mode, ndims=2)
    v_coord, u_coord = plot_defn.coords

    # Find which plot coordinate corresponds to the derived altitude, so that we
    # can replace altitude with the surface altitude
    if v_coord and v_coord.standard_name == 'altitude':
        # v is altitude, so plot u and orography with orog in the y direction
        result = vert_plot(u_coord, orography, style_args)
    elif u_coord and u_coord.standard_name == 'altitude':
        # u is altitude, so plot v and orography with orog in the x direction
        result = horiz_plot(v_coord, orography, style_args)
    else:
        raise ValueError('Plot does not use hybrid height. One of the coordinates '
                         'to plot must be altitude, but %s and %s were '
                         'given.' % (u_coord.name(), v_coord.name()))
    return result


def orography_at_bounds(cube, facecolor='#888888', coords=None):
    """Plots orography defined at cell boundaries from the given Cube."""

    # XXX Needs contiguous orography corners to work.
    raise NotImplementedError('This operation is temporarily not provided until ' 
                              'coordinates can expose 2d contiguous bounds (corners).')
    
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
        
    return _fill_orography(cube, coords, iris.coords.BOUND_MODE, vert_plot, horiz_plot, style_args)


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

    return _fill_orography(cube, coords, iris.coords.POINT_MODE, vert_plot, horiz_plot, style_args)


def outline(cube, coords=None):
    """
    Draws cell outlines based on the given Cube.
    
    Args:
    
    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the given coordinates
        indicates which axis to use for each, where the first element is the horizontal axis of
        the plot and the second element is the vertical axis of the plot.
        
    """
    result = _draw_2d_from_bounds('pcolormesh', cube, facecolors='none', edgecolors='k', antialiased=True, coords=coords)
    # set the _is_stroked property to get a single color grid. See https://github.com/matplotlib/matplotlib/issues/1302
    result._is_stroked = False
    return result


@iris.palette.auto_palette
def pcolor(cube, *args, **kwargs):
    """
    Draws a pseudocolor plot based on the given Cube.
    
    Args:
    
    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the given coordinates
        indicates which axis to use for each, where the first element is the horizontal axis of
        the plot and the second element is the vertical axis of the plot.
        
    See :func:`matplotlib.pyplot.pcolor` for details of other valid keyword arguments.
    
    """
    kwargs.setdefault('antialiased', True)
    result = _draw_2d_from_bounds('pcolor', cube, *args, **kwargs)
    if iris.palette.is_brewer(kwargs.get('cmap')):
        citation(_BREWER)
    return result


@iris.palette.auto_palette
def pcolormesh(cube, *args, **kwargs):
    """
    Draws a pseudocolor plot based on the given Cube.
    
    Args:
    
    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the given coordinates
        indicates which axis to use for each, where the first element is the horizontal axis of
        the plot and the second element is the vertical axis of the plot.
    
    See :func:`matplotlib.pyplot.pcolormesh` for details of other valid keyword arguments.
    
    """
    result = _draw_2d_from_bounds('pcolormesh', cube, *args, **kwargs)
    if iris.palette.is_brewer(kwargs.get('cmap')):
        citation(_BREWER)
    return result


def points(cube, *args, **kwargs):
    """
    Draws sample point positions based on the given Cube.
    
    Args:
    
    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the given coordinates
        indicates which axis to use for each, where the first element is the horizontal axis of
        the plot and the second element is the vertical axis of the plot.
        
    See :func:`matplotlib.pyplot.scatter` for details of other valid keyword arguments.
    
    """
    _scatter_args = lambda u, v, data, *args, **kwargs: \
                        ((u.astype(numpy.double), v.astype(numpy.double)) + args, kwargs)
    return _draw_2d_from_points('scatter', _scatter_args, cube, *args, **kwargs)


def plot(cube, *args, **kwargs):
    """
    Draws a line plot based on the given Cube.
    
    Args:
    
    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the given coordinates
        indicates which axis to use for each, where the first element is the horizontal axis of
        the plot and the second element is the vertical axis of the plot. 
    
    See :func:`matplotlib.pyplot.plot` for details of other valid keyword arguments.
    
    """
    _plot_args = None
    return _draw_1d_from_points('plot', _plot_args, cube, *args, **kwargs)


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
        The :class:`matplotlib.axes.Axes` in which the symbols will be added. Defaults to the current axes.
    
    * units: ['inches', 'points']
        The unit for the symbol size.
    
    """
    if axes is None:
        axes = plt.gca()
        
    offsets = numpy.array(zip(x, y))
    
    # XXX "match_original" doesn't work ... so brute-force it instead.
    #   PatchCollection constructor ignores all non-style keywords when using match_original
    #   See matplotlib.collections.PatchCollection.__init__
    #   Specifically matplotlib/collections line 1053
    #pc = PatchCollection(symbols, offsets=offsets, transOffset=ax.transData, match_original=True)
    facecolors = [p.get_facecolor() for p in symbols]
    edgecolors = [p.get_edgecolor() for p in symbols]
    linewidths = [p.get_linewidth() for p in symbols]
    
    pc = mpl_collections.PatchCollection(symbols, offsets=offsets, transOffset=axes.transData,
            facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths)
            
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

