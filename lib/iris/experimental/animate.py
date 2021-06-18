# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Wrapper for animating iris cubes using iris or matplotlib plotting functions

"""

import warnings

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import iris


def animate(cube_iterator, plot_func, fig=None, **kwargs):
    """
    Animates the given cube iterator.

    Args:

    * cube_iterator (iterable of :class:`iris.cube.Cube` objects):
        Each animation frame corresponds to each :class:`iris.cube.Cube`
        object. See :meth:`iris.cube.Cube.slices`.

    * plot_func (:mod:`iris.plot` or :mod:`iris.quickplot` plotting function):
        Plotting function used to animate. Must accept the signature
        ``plot_func(cube, vmin=vmin, vmax=vmax, coords=coords)``.
        :func:`~iris.plot.contourf`, :func:`~iris.plot.contour`,
        :func:`~iris.plot.pcolor` and :func:`~iris.plot.pcolormesh`
        all conform to this signature.

    Kwargs:

    * fig (:class:`matplotlib.figure.Figure` instance):
        By default, the current figure will be used or a new figure instance
        created if no figure is available. See :func:`matplotlib.pyplot.gcf`.

    * coords (list of :class:`~iris.coords.Coord` objects or coordinate names):
        Use the given coordinates as the axes for the plot. The order of the
        given coordinates indicates which axis to use for each, where the first
        element is the horizontal axis of the plot and the second element is
        the vertical axis of the plot.

    * interval (int, float or long):
        Defines the time interval in milliseconds between successive frames.
        A default interval of 100ms is set.

    * vmin, vmax (int, float or long):
        Color scaling values, see :class:`matplotlib.colors.Normalize` for
        further details. Default values are determined by the min-max across
        the data set over the entire sequence.

    See :class:`matplotlib.animation.FuncAnimation` for details of other valid
    keyword arguments.

    Returns:
        :class:`~matplotlib.animation.FuncAnimation` object suitable for
        saving and or plotting.

    For example, to animate along a set of cube slices::

        cube_iter = cubes.slices(('grid_longitude', 'grid_latitude'))
        ani = animate(cube_iter, qplt.contourf)
        plt.show()

    """
    kwargs.setdefault("interval", 100)
    coords = kwargs.pop("coords", None)

    if fig is None:
        fig = plt.gcf()

    def update_animation_iris(i, cubes, vmin, vmax, coords):
        # Clearing the figure is currently necessary for compatibility with
        # the iris quickploting module - due to the colorbar.
        plt.gcf().clf()
        plot_func(cubes[i], vmin=vmin, vmax=vmax, coords=coords)

    # Turn cube iterator into a list to determine plot ranges.
    # NOTE: we check that we are not providing a cube as this has a deprecated
    # iter special method.
    if hasattr(cube_iterator, "__iter__") and not isinstance(
        cube_iterator, iris.cube.Cube
    ):
        cubes = iris.cube.CubeList(cube_iterator)
    else:
        msg = "iterable type object required for animation, {} given".format(
            type(cube_iterator)
        )
        raise TypeError(msg)

    supported = ["iris.plot", "iris.quickplot"]
    if plot_func.__module__ not in supported:
        msg = (
            'Given plotting module "{}" may not be supported, intended '
            "use: {}."
        )
        msg = msg.format(plot_func.__module__, supported)
        warnings.warn(msg, UserWarning)

    supported = ["contour", "contourf", "pcolor", "pcolormesh"]
    if plot_func.__name__ not in supported:
        msg = (
            'Given plotting function "{}" may not be supported, intended '
            "use: {}."
        )
        msg = msg.format(plot_func.__name__, supported)
        warnings.warn(msg, UserWarning)

    # Determine plot range.
    vmin = kwargs.pop("vmin", min([cc.data.min() for cc in cubes]))
    vmax = kwargs.pop("vmax", max([cc.data.max() for cc in cubes]))

    update = update_animation_iris
    frames = range(len(cubes))

    return animation.FuncAnimation(
        fig, update, frames=frames, fargs=(cubes, vmin, vmax, coords), **kwargs
    )
