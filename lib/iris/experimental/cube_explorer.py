# (C) British Crown Copyright 2013, Met Office
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
Class to allow interactive exploration of a cube, including:
    * buttons to navigate through non-displayed dimensions.
    * buttons to animate through non-displayed dimensions.
    * a picker to make pop-up plots of data in non-displayed
        dimensions at select points.

"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from iris.exceptions import CoordinateNotFoundError


class ButtonSetup(object):
    """
    Convenience class to store setup for Cube
    Explorer buttons.

    Args:

    * col_margin:
        Value for setting the amount of space
        on the right of the plot for displaying
        buttons.

    * initial_but_a_pos:
        Positions of button a axis
        in slot one.

    * initial_but_b_pos:
        Positions of button b axis
        in slot one.

    * slot_diff:
        The vertical offset between
        button slots.

    """

    def __init__(self,
                 col_margin=0.85,
                 initial_but_a_pos=[0.875, 0.85, 0.11, 0.05],
                 initial_but_b_pos=[0.875, 0.79, 0.11, 0.05],
                 slot_diff=0.15):
        self.col_margin = col_margin
        self.initial_but_a_pos = initial_but_a_pos
        self.initial_but_b_pos = initial_but_b_pos
        self.slot_diff = slot_diff


class CubeExplorer(object):
    def __init__(self, cube, plot_func, current_slice, *args, **kwargs):
        """
        Args:

        * cube:
            Cube of data to plot.

        * plot_func:
            Pointer to a plotting function
            which is compatible with the slice defined
            by current_slice.

        * current_slice:
            Index tuple which gives a slice of cube
            which is compatible with plot_func.

        Kwargs:

        * hook:
            A function that accepts the cube explorer
            object to allow formatting of the plot.

        """

        self.cube = cube
        self.plot_func = plot_func
        self.current_slice = current_slice
        self.hook = kwargs.pop('hook', None)
        self.button_setup = kwargs.pop('button_setup',
                                       ButtonSetup())
        # self.ax is set when the plot is made
        self.ax = None
        self._plot_args = args
        self._plot_kwargs = kwargs
        # _butts and _butt_fns are added to with
        # various class methods
        self._butts = {}
        self._butt_fns = {}

        self._make_plot()

    def show(self):
        plt.show()

    def add_nav_buttons(self, dim, names_tup, slot=0, circular=False):
        """
        Adds a set of two buttons to the plot window which allow incrementing
        or decrementing over the specified dimension

        Args:

        * dim:
            Dimension number or name to traverse.

        * names_tup:
            Tuple of two strings to be the names of the
            increment and decrement buttons respectively.

        * slot:
            Level of the plot to display this button set.

        * circular:
            Boolean - to loop round when limit is reached or not.

        """
        if type(dim) is str:
            dim, = self.cube.coord_dims(self.cube.coord(dim))

        if type(self.current_slice[dim]) is slice:
            raise TypeError("Cannot iterate over a displayed dimension")

        butt_funcs_tup = (self._get_nav_fn(dim, 'inc', circular),
                          self._get_nav_fn(dim, 'dec', circular))
        self._add_butt_pair(names_tup, butt_funcs_tup, slot)

    def add_color_range_buttons(self, names_tup, delta, slot):
        """
        Adds two buttons to expand/contract the displayed
        color range.

        Args:

        * name_tup
            A tuple of two srtings to be the names of the
            expand and collapse range buttons respectively.

        * delta
            The amount to expand/contract each end of the
            color range i.e. a delta of x will lower vmin
            by x and increase vmax by x, increasing the
            range by 2x.

        * slot
            Level of the plot to display this button set

        """
        if "vmin" in self._plot_kwargs and "vmax" in self._plot_kwargs:
            def inc_col(event):
                self._plot_kwargs["vmax"] += delta
                self._plot_kwargs["vmin"] -= delta
                self._refresh_plot()

            def dec_col(event):
                self._plot_kwargs["vmax"] -= delta
                self._plot_kwargs["vmin"] += delta
                self._refresh_plot()

            self._add_butt_pair(names_tup,
                                (inc_col, dec_col),
                                slot)
        else:
            raise KeyError("Please set initial vmin and vmax values when"
                           " initialising the cube_explorer.")

    def add_animate_buttons(self, dim, names_tup, slot=0, refresh_rate=0.2):
        """
        Adds a set of two buttons to start/stop cycling through a dimension
        of a cube.

        Args:

        * dim:
            Dimension number/name to traverse.

        * names_tup:
            Tuple of two strings to be the names of the
            play and stop buttons respectively.

        * slot:
            Level of the plot to display this button set.

        * refresh_rate:
            Number of seconds to wait between each frame.

        """
        if type(dim) is str:
            dim, = self.cube.coord_dims(self.cube.coord(dim))

        if type(self.current_slice[dim]) is slice:
            raise TypeError("Cannot iterate over a displayed dimension")

        butt_funcs_tup = self._get_ani_fns(dim, refresh_rate)
        self._add_butt_pair(names_tup, butt_funcs_tup, slot)

    def add_picker(self, plot_func, *args, **kwargs):
        """
        Adds picker functionality to a Cube Explorer object.
        The user can then select points from which to display
        data in one of the non-displayed dimension in a pop
        up plot by clicking on the cube explorer plot.

        Args:

        * plot_func:
            A plotting function that accepts a cube
            consisting of all the non-displayed dimensions.

        Kwargs:

        * hook:
            A function that accepts the axes object of the
            pop up plot, and the cube explorer object to
            allow formatting of the pop up plot.

        """
        hook = kwargs.pop('hook', None)
        # add picker option to plot
        self._plot_kwargs['picker'] = kwargs.pop('picker', True)
        self._refresh_plot()

        def _on_pick(event):
            if event.artist.get_axes() != self.ax:
                return

            s = [slice(None)] * len(self.cube.shape)
            picker_plot_dims = [i for i, v in
                                enumerate(self.current_slice)
                                if type(v) is slice]

            xpt = event.mouseevent.xdata
            s[picker_plot_dims[0]] = int(xpt)
            if len(picker_plot_dims) == 2:
                ypt = event.mouseevent.ydata
                s[picker_plot_dims[1]] = int(ypt)

            plt.figure(num=None)
            plot_func(self.cube[tuple(s)], *args, **kwargs)
            if hook is not None:
                popup_ax = plt.gca()
                hook(popup_ax, self)
            plt.show()

        self.ax.figure.canvas.mpl_connect('pick_event', _on_pick)

    def _add_butt_pair(self, names_tup, butt_funcs_tup, slot):
        """
        Assigns two functions to two buttons

        """
        plt.subplots_adjust(right=self.button_setup.col_margin)

        but_pos_a = self.button_setup.initial_but_a_pos
        but_pos_a[1] -= slot*self.button_setup.slot_diff
        self._butts[names_tup[0]] = Button(plt.axes(but_pos_a), names_tup[0])
        self._butt_fns[names_tup[0]] = butt_funcs_tup[0]
        self._butts[names_tup[0]].on_clicked(self._butt_fns[names_tup[0]])

        but_pos_b = self.button_setup.initial_but_b_pos
        but_pos_b[1] -= slot*self.button_setup.slot_diff
        self._butts[names_tup[1]] = Button(plt.axes(but_pos_b), names_tup[1])
        self._butt_fns[names_tup[1]] = butt_funcs_tup[1]
        self._butts[names_tup[1]].on_clicked(self._butt_fns[names_tup[1]])

        # set axis back to plot
        plt.sca(self.ax)

    def _make_plot(self):
        """
        Makes initial plot

        """
        self.fig = plt.figure(num=None)

        # Make the initial plot.
        self.pl = self.plot_func(self.cube[tuple(self.current_slice)],
                                 *self._plot_args,
                                 **self._plot_kwargs)
        self.ax = plt.gca()

        if self.hook is not None:
            self.hook(self)

    def _refresh_plot(self):
        """
        Refreshes the displayed plot to display the slice defined
        by current_slice.

        """

        self.ax.clear()
        self.plot_func(self.cube[tuple(self.current_slice)],
                       *self._plot_args,
                       **self._plot_kwargs)
        if self.hook is not None:
            self.hook(self)
        self.fig.canvas.draw()

    def _get_nav_fn(self, dim, inc_or_dec, circular):
        """
        Returns increment and decrement button functions for a dimension.

        """
        if inc_or_dec is 'inc':
            def fn(event):
                if self.current_slice[dim] < self.cube.shape[dim]-1:
                    self.current_slice[dim] += 1
                elif circular:
                    self.current_slice[dim] = 0

                self._refresh_plot()

        elif inc_or_dec is 'dec':
            def fn(event):
                if self.current_slice[dim] > 0:
                    self.current_slice[dim] -= 1
                elif circular:
                    self.current_slice[dim] = self.cube.shape[dim]

                self._refresh_plot()

        return fn

    def _get_ani_fns(self, dim, refresh_rate):
        def play(event):
            self.playing = True
            while True:
                self.fig.canvas.start_event_loop(timeout=refresh_rate)
                if self.playing:
                    if self.current_slice[dim] < self.cube.shape[dim]-1:
                        self.current_slice[dim] += 1
                    else:
                        self.current_slice[dim] = 0
                    self._refresh_plot()
                else:
                    self._refresh_plot()
                    break

        def stop(event):
            self.playing = False

        return play, stop
