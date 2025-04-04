# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.plot` module."""

import pytest

from iris.coords import AuxCoord
from iris.plot import _broadcast_2d as broadcast
from iris.tests import _shared_utils
from iris.tests.stock import lat_lon_cube, simple_2d


@_shared_utils.skip_plot
class TestGraphicStringCoord(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = simple_2d(with_bounds=True)
        self.cube.add_aux_coord(AuxCoord(list("abcd"), long_name="str_coord"), 1)
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

    def assert_bounds_tick_labels(self, axis, axes=None):
        actual = self.tick_loc_and_label(axis, axes)
        expected = [
            (-1.0, ""),
            (0.0, "a"),
            (1.0, "b"),
            (2.0, "c"),
            (3.0, "d"),
            (4.0, ""),
        ]
        assert expected == actual

    def assert_points_tick_labels(self, axis, axes=None):
        actual = self.tick_loc_and_label(axis, axes)
        expected = [(0.0, "a"), (1.0, "b"), (2.0, "c"), (3.0, "d")]
        assert expected == actual


@_shared_utils.skip_plot
class MixinCoords:
    """Mixin class of common plotting tests providing 2-dimensional
    permutations of coordinates and anonymous dimensions.

    """

    def _check(self, u, v, data=None):
        assert self.mpl_patch.call_count == 1
        if data is not None:
            (actual_u, actual_v, actual_data), _ = self.mpl_patch.call_args
            _shared_utils.assert_array_equal(actual_data, data)
        else:
            (actual_u, actual_v), _ = self.mpl_patch.call_args
        _shared_utils.assert_array_equal(actual_u, u)
        _shared_utils.assert_array_equal(actual_v, v)

    def test_foo_bar(self):
        self.draw_func(self.cube, coords=("foo", "bar"))
        u, v = broadcast(self.foo, self.bar)
        self._check(u, v, self.data)

    def test_bar_foo(self):
        self.draw_func(self.cube, coords=("bar", "foo"))
        u, v = broadcast(self.bar, self.foo)
        self._check(u, v, self.dataT)

    def test_foo_0(self):
        self.draw_func(self.cube, coords=("foo", 0))
        u, v = broadcast(self.foo, self.bar_index)
        self._check(u, v, self.data)

    def test_1_bar(self):
        self.draw_func(self.cube, coords=(1, "bar"))
        u, v = broadcast(self.foo_index, self.bar)
        self._check(u, v, self.data)

    def test_1_0(self):
        self.draw_func(self.cube, coords=(1, 0))
        u, v = broadcast(self.foo_index, self.bar_index)
        self._check(u, v, self.data)

    def test_0_foo(self):
        self.draw_func(self.cube, coords=(0, "foo"))
        u, v = broadcast(self.bar_index, self.foo)
        self._check(u, v, self.dataT)

    def test_bar_1(self):
        self.draw_func(self.cube, coords=("bar", 1))
        u, v = broadcast(self.bar, self.foo_index)
        self._check(u, v, self.dataT)

    def test_0_1(self):
        self.draw_func(self.cube, coords=(0, 1))
        u, v = broadcast(self.bar_index, self.foo_index)
        self._check(u, v, self.dataT)
