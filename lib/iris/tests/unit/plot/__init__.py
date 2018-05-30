# (C) British Crown Copyright 2014 - 2018, Met Office
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
"""Unit tests for the :mod:`iris.plot` module."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.plot import _broadcast_2d as broadcast
from iris.coords import AuxCoord
from iris.tests.stock import simple_2d, lat_lon_cube


@tests.skip_plot
class TestGraphicStringCoord(tests.GraphicsTest):
    def setUp(self):
        super(TestGraphicStringCoord, self).setUp()
        self.cube = simple_2d(with_bounds=True)
        self.cube.add_aux_coord(AuxCoord(list('abcd'),
                                         long_name='str_coord'), 1)
        self.lat_lon_cube = lat_lon_cube()

    def tick_loc_and_label(self, axis_name, axes=None):
        # Intentional lazy import so that subclasses can have an opportunity
        # to change the backend.
        import matplotlib.pyplot as plt

        # Draw the plot to 'fix' the ticks.
        if axes:
            axes.figure.canvas.draw()
        else:
            axes = plt.gca()
            plt.draw()
        axis = getattr(axes, axis_name)

        locations = axis.get_majorticklocs()
        labels = [tick.get_text() for tick in axis.get_ticklabels()]
        return list(zip(locations, labels))

    def assertBoundsTickLabels(self, axis, axes=None):
        actual = self.tick_loc_and_label(axis, axes)
        expected = [(-1.0, 'a'), (0.0, 'a'), (1.0, 'b'),
                    (2.0, 'c'), (3.0, 'd'), (4.0, u'')]
        self.assertEqual(expected, actual)

    def assertPointsTickLabels(self, axis, axes=None):
        actual = self.tick_loc_and_label(axis, axes)
        expected = [(0.0, 'a'), (1.0, 'b'), (2.0, 'c'), (3.0, 'd')]
        self.assertEqual(expected, actual)


@tests.skip_plot
class MixinCoords(object):
    """
    Mixin class of common plotting tests providing 2-dimensional
    permutations of coordinates and anonymous dimensions.

    """
    def _check(self, u, v, data=None):
        self.assertEqual(self.mpl_patch.call_count, 1)
        if data is not None:
            (actual_u, actual_v, actual_data), _ = self.mpl_patch.call_args
            self.assertArrayEqual(actual_data, data)
        else:
            (actual_u, actual_v), _ = self.mpl_patch.call_args
        self.assertArrayEqual(actual_u, u)
        self.assertArrayEqual(actual_v, v)

    def test_foo_bar(self):
        self.draw_func(self.cube, coords=('foo', 'bar'))
        u, v = broadcast(self.foo, self.bar)
        self._check(u, v, self.data)

    def test_bar_foo(self):
        self.draw_func(self.cube, coords=('bar', 'foo'))
        u, v = broadcast(self.bar, self.foo)
        self._check(u, v, self.dataT)

    def test_foo_0(self):
        self.draw_func(self.cube, coords=('foo', 0))
        u, v = broadcast(self.foo, self.bar_index)
        self._check(u, v, self.data)

    def test_1_bar(self):
        self.draw_func(self.cube, coords=(1, 'bar'))
        u, v = broadcast(self.foo_index, self.bar)
        self._check(u, v, self.data)

    def test_1_0(self):
        self.draw_func(self.cube, coords=(1, 0))
        u, v = broadcast(self.foo_index, self.bar_index)
        self._check(u, v, self.data)

    def test_0_foo(self):
        self.draw_func(self.cube, coords=(0, 'foo'))
        u, v = broadcast(self.bar_index, self.foo)
        self._check(u, v, self.dataT)

    def test_bar_1(self):
        self.draw_func(self.cube, coords=('bar', 1))
        u, v = broadcast(self.bar, self.foo_index)
        self._check(u, v, self.dataT)

    def test_0_1(self):
        self.draw_func(self.cube, coords=(0, 1))
        u, v = broadcast(self.bar_index, self.foo_index)
        self._check(u, v, self.dataT)
