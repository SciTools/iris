# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot.contour` function."""

import numpy as np
import pytest

from iris.tests import _shared_utils
from iris.tests.stock import simple_2d
from iris.tests.unit.plot import MixinCoords, TestGraphicStringCoord

if _shared_utils.MPL_AVAILABLE:
    import iris.plot as iplt


@_shared_utils.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def test_yaxis_labels(self):
        iplt.contour(self.cube, coords=("bar", "str_coord"))
        self.assert_points_tick_labels("yaxis")

    def test_xaxis_labels(self):
        iplt.contour(self.cube, coords=("str_coord", "bar"))
        self.assert_points_tick_labels("xaxis")

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.contour(self.cube, axes=ax, coords=("bar", "str_coord"))
        plt.close(fig)
        self.assert_points_tick_labels("yaxis", ax)

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.contour(self.cube, axes=ax, coords=("str_coord", "bar"))
        plt.close(fig)
        self.assert_points_tick_labels("xaxis", ax)

    def test_geoaxes_exception(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        pytest.raises(TypeError, iplt.contour, self.lat_lon_cube, axes=ax)
        plt.close(fig)


@_shared_utils.skip_plot
class TestCoords(MixinCoords):
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        # We have a 2d cube with dimensionality (bar: 3; foo: 4)
        self.cube = simple_2d(with_bounds=False)
        self.foo = self.cube.coord("foo").points
        self.foo_index = np.arange(self.foo.size)
        self.bar = self.cube.coord("bar").points
        self.bar_index = np.arange(self.bar.size)
        self.data = self.cube.data
        self.dataT = self.data.T
        self.mpl_patch = mocker.patch("matplotlib.pyplot.contour")
        self.draw_func = iplt.contour
