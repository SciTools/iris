# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.quickplot.plot` function."""

import pytest

from iris.tests import _shared_utils
from iris.tests.stock import simple_1d
from iris.tests.unit.plot import TestGraphicStringCoord

if _shared_utils.MPL_AVAILABLE:
    import iris.quickplot as qplt


@_shared_utils.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    parent_setup = TestGraphicStringCoord._setup

    @pytest.fixture(autouse=True)
    def _setup(self, parent_setup):
        self.cube = self.cube[0, :]

    def test_yaxis_labels(self):
        qplt.plot(self.cube, self.cube.coord("str_coord"))
        self.assert_bounds_tick_labels("yaxis")

    def test_xaxis_labels(self):
        qplt.plot(self.cube.coord("str_coord"), self.cube)
        self.assert_bounds_tick_labels("xaxis")


class TestAxisLabels(_shared_utils.GraphicsTest):
    def test_xy_cube(self):
        c = simple_1d()
        qplt.plot(c)
        ax = qplt.plt.gca()
        x = ax.xaxis.get_label().get_text()
        assert x == "Foo"
        y = ax.yaxis.get_label().get_text()
        assert y == "Thingness"

    def test_yx_cube(self):
        c = simple_1d()
        c.transpose()
        # Making the cube a vertical coordinate should change the default
        # orientation of the plot.
        c.coord("foo").attributes["positive"] = "up"
        qplt.plot(c)
        ax = qplt.plt.gca()
        x = ax.xaxis.get_label().get_text()
        assert x == "Thingness"
        y = ax.yaxis.get_label().get_text()
        assert y == "Foo"
