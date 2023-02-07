# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
High-level plotting extensions to :mod:`iris.plot`.

These routines work much like their :mod:`iris.plot` counterparts, but they
automatically add a plot title, axis titles, and a colour bar when appropriate.

See also: :ref:`matplotlib <matplotlib:users-guide-index>`.

"""

import cf_units
from matplotlib import __version__ as _mpl_version
import matplotlib.pyplot as plt
from packaging import version

import iris.config
import iris.coords
import iris.plot as iplt


def _use_symbol(units):
    # For non-time units use the shortest unit representation.
    # E.g. prefer 'K' over 'kelvin', but not '0.0174532925199433 rad'
    # over 'degrees'
    return (
        not units.is_time()
        and not units.is_time_reference()
        and len(units.symbol) < len(str(units))
    )


def _title(cube_or_coord, with_units):
    if cube_or_coord is None or isinstance(cube_or_coord, int):
        title = ""
    else:
        title = cube_or_coord.name().replace("_", " ").capitalize()
        units = cube_or_coord.units
        if with_units and not (
            units.is_unknown()
            or units.is_no_unit()
            or units == cf_units.Unit("1")
        ):
            if _use_symbol(units):
                units = units.symbol
            elif units.is_time_reference():
                # iris.plot uses matplotlib.dates.date2num, which is fixed to the below unit.
                if version.parse(_mpl_version) >= version.parse("3.3"):
                    days_since = "1970-01-01"
                else:
                    days_since = "0001-01-01"
                units = "days since {}".format(days_since)
            title += " / {}".format(units)

    return title


def _label(cube, mode, result=None, ndims=2, coords=None, axes=None):
    """Puts labels on the current plot using the given cube."""

    if axes is None:
        axes = plt.gca()

    axes.set_title(_title(cube, with_units=False))

    if result is not None:
        draw_edges = mode == iris.coords.POINT_MODE
        bar = plt.colorbar(
            result, ax=axes, orientation="horizontal", drawedges=draw_edges
        )
        has_known_units = not (
            cube.units.is_unknown() or cube.units.is_no_unit()
        )
        if has_known_units and cube.units != cf_units.Unit("1"):
            # Use shortest unit representation for anything other than time
            if _use_symbol(cube.units):
                bar.set_label(cube.units.symbol)
            else:
                bar.set_label(cube.units)
        # Remove the tick which is put on the colorbar by default.
        bar.ax.tick_params(length=0)

    if coords is None:
        plot_defn = iplt._get_plot_defn(cube, mode, ndims)
    else:
        plot_defn = iplt._get_plot_defn_custom_coords_picked(
            cube, coords, mode, ndims=ndims
        )

    if ndims == 2:
        if not iplt._can_draw_map(plot_defn.coords):
            axes.set_ylabel(_title(plot_defn.coords[0], with_units=True))
            axes.set_xlabel(_title(plot_defn.coords[1], with_units=True))
    elif ndims == 1:
        axes.set_xlabel(_title(plot_defn.coords[0], with_units=True))
        axes.set_ylabel(_title(cube, with_units=True))
    else:
        msg = (
            "Unexpected number of dimensions ({}) given to "
            "_label.".format(ndims)
        )
        raise ValueError(msg)


def _label_with_bounds(cube, result=None, ndims=2, coords=None, axes=None):
    _label(cube, iris.coords.BOUND_MODE, result, ndims, coords, axes)


def _label_with_points(cube, result=None, ndims=2, coords=None, axes=None):
    _label(cube, iris.coords.POINT_MODE, result, ndims, coords, axes)


def _get_titles(u_object, v_object):
    if u_object is None:
        u_object = iplt._u_object_from_v_object(v_object)
    xunits = u_object is not None and not u_object.units.is_time_reference()
    yunits = not v_object.units.is_time_reference()
    xlabel = _title(u_object, with_units=xunits)
    ylabel = _title(v_object, with_units=yunits)
    title = ""
    if u_object is None:
        title = _title(v_object, with_units=False)
    elif isinstance(u_object, iris.cube.Cube) and not isinstance(
        v_object, iris.cube.Cube
    ):
        title = _title(u_object, with_units=False)
    elif isinstance(v_object, iris.cube.Cube) and not isinstance(
        u_object, iris.cube.Cube
    ):
        title = _title(v_object, with_units=False)
    return xlabel, ylabel, title


def _label_1d_plot(*args, **kwargs):
    u_obj, v_obj, _, _, _ = iplt._get_plot_objects(args)
    xlabel, ylabel, title = _get_titles(u_obj, v_obj)

    axes = kwargs.pop("axes", None)

    if len(kwargs) != 0:
        msg = "Unexpected kwargs {} given to _label_1d_plot".format(
            kwargs.keys()
        )
        raise ValueError(msg)

    if axes is None:
        axes = plt.gca()

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)


def contour(cube, *args, **kwargs):
    """
    Draws contour lines on a labelled plot based on the given Cube.

    With the basic call signature, contour "level" values are chosen
    automatically::

        contour(cube)

    Supply a number to use *N* automatically chosen levels::

        contour(cube, N)

    Supply a sequence *V* to use explicitly defined levels::

        contour(cube, V)

    See :func:`iris.plot.contour` for details of valid keyword arguments.

    """
    coords = kwargs.get("coords")
    axes = kwargs.get("axes")
    result = iplt.contour(cube, *args, **kwargs)
    _label_with_points(cube, coords=coords, axes=axes)
    return result


def contourf(cube, *args, **kwargs):
    """
    Draws filled contours on a labelled plot based on the given Cube.

    With the basic call signature, contour "level" values are chosen
    automatically::

        contour(cube)

    Supply a number to use *N* automatically chosen levels::

        contour(cube, N)

    Supply a sequence *V* to use explicitly defined levels::

        contour(cube, V)

    See :func:`iris.plot.contourf` for details of valid keyword arguments.

    """
    coords = kwargs.get("coords")
    axes = kwargs.get("axes")
    result = iplt.contourf(cube, *args, **kwargs)
    _label_with_points(cube, result, coords=coords, axes=axes)
    return result


def outline(cube, coords=None, color="k", linewidth=None, axes=None):
    """
    Draws cell outlines on a labelled plot based on the given Cube.

    Kwargs:

    * coords: list of :class:`~iris.coords.Coord` objects or coordinate names
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    * color: None or mpl color
        The color of the cell outlines. If None, the matplotlibrc setting
        patch.edgecolor is used by default.

    * linewidth: None or number
        The width of the lines showing the cell outlines. If None, the default
        width in patch.linewidth in matplotlibrc is used.

    """
    result = iplt.outline(
        cube, color=color, linewidth=linewidth, coords=coords, axes=axes
    )

    _label_with_bounds(cube, coords=coords, axes=axes)
    return result


def pcolor(cube, *args, **kwargs):
    """
    Draws a labelled pseudocolor plot based on the given Cube.

    See :func:`iris.plot.pcolor` for details of valid keyword arguments.

    """
    coords = kwargs.get("coords")
    axes = kwargs.get("axes")
    result = iplt.pcolor(cube, *args, **kwargs)
    _label_with_bounds(cube, result, coords=coords, axes=axes)
    return result


def pcolormesh(cube, *args, **kwargs):
    """
    Draws a labelled pseudocolour plot based on the given Cube.

    See :func:`iris.plot.pcolormesh` for details of valid keyword arguments.

    """
    coords = kwargs.get("coords")
    axes = kwargs.get("axes")
    result = iplt.pcolormesh(cube, *args, **kwargs)
    _label_with_bounds(cube, result, coords=coords, axes=axes)
    return result


def points(cube, *args, **kwargs):
    """
    Draws sample point positions on a labelled plot based on the given Cube.

    See :func:`iris.plot.points` for details of valid keyword arguments.

    """
    coords = kwargs.get("coords")
    axes = kwargs.get("axes")
    result = iplt.points(cube, *args, **kwargs)
    _label_with_points(cube, coords=coords, axes=axes)
    return result


def plot(*args, **kwargs):
    """
    Draws a labelled line plot based on the given cube(s) or
    coordinate(s).

    See :func:`iris.plot.plot` for details of valid arguments and
    keyword arguments.

    """
    axes = kwargs.get("axes")
    result = iplt.plot(*args, **kwargs)
    _label_1d_plot(*args, axes=axes)
    return result


def scatter(x, y, *args, **kwargs):
    """
    Draws a labelled scatter plot based on the given cubes or
    coordinates.

    See :func:`iris.plot.scatter` for details of valid arguments and
    keyword arguments.

    """
    axes = kwargs.get("axes")
    result = iplt.scatter(x, y, *args, **kwargs)
    _label_1d_plot(x, y, axes=axes)
    return result


def fill_between(x, y1, y2, *args, **kwargs):
    """
    Draws a labelled fill_between plot based on the given cubes or coordinates.

    See :func:`iris.plot.fill_between` for details of valid arguments and
    keyword arguments.

    """
    axes = kwargs.get("axes")
    result = iplt.fill_between(x, y1, y2, *args, **kwargs)
    _label_1d_plot(x, y1, axes=axes)
    return result


# Provide a convenience show method from pyplot.
show = plt.show
