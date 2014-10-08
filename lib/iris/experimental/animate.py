# (C) British Crown Copyright 2013 - 2014, Met Office
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
Wrapper for animating iris cubes using iris or matplotlib plotting functions

"""

from __future__ import (absolute_import, division, print_function)

import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import iris


def animate(cube_iterator, plot_func, fig=None, **kwargs):
    """
    Animates the given cube iterator.

    Args:

    * cube_iterator (iterable of :class:`iris.cube.Cube` objects):
        Each animation frame corresponds to each :class:`iris.cube.Cube`
        object. See :meth:`iris.cube.Cube.slices`.

    * plot_func (:mod:`~iris.plot` or :mod:`~iris.quickplot` plot):
        Plotting function used to animate.

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
    kwargs.setdefault('interval', 100)
    coords = kwargs.pop('coords', None)

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
    if (hasattr(cube_iterator, '__iter__') and not
            isinstance(cube_iterator, iris.cube.Cube)):
        cubes = iris.cube.CubeList(cube_iterator)
    else:
        msg = 'iterable type object required for animation, {} given'.format(
            type(cube_iterator))
        raise TypeError(msg)

    supported = ['iris.plot', 'iris.quickplot']
    if plot_func.__module__ not in supported:
        msg = ('Given plotting module "{}" may not be supported, intended '
               'use: {}.')
        msg = msg.format(plot_func.__module__, supported)
        warnings.warn(msg, UserWarning)

    supported = ['contour', 'contourf', 'pcolor', 'pcolormesh']
    if plot_func.__name__ not in supported:
        msg = ('Given plotting function "{}" may not be supported, intended '
               'use: {}.')
        msg = msg.format(plot_func.__name__, supported)
        warnings.warn(msg, UserWarning)

    # Determine plot range.
    vmin = kwargs.pop('vmin', min([cc.data.min() for cc in cubes]))
    vmax = kwargs.pop('vmax', max([cc.data.max() for cc in cubes]))

    update = update_animation_iris
    frames = xrange(len(cubes))

    return animation.FuncAnimation(fig, update,
                                   frames=frames,
                                   fargs=(cubes, vmin, vmax, coords),
                                   **kwargs)
