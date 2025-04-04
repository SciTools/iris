# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot.scatter` function."""

import pytest

from iris.tests import _shared_utils
from iris.tests.unit.plot import TestGraphicStringCoord

if _shared_utils.MPL_AVAILABLE:
    import iris.plot as iplt


@_shared_utils.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    parent_setup = TestGraphicStringCoord._setup

    @pytest.fixture(autouse=True)
    def _setup(self, parent_setup):
        self.cube = self.cube[0, :]
        self.lat_lon_cube = self.lat_lon_cube[0, :]

    def test_xaxis_labels(self):
        iplt.scatter(self.cube.coord("str_coord"), self.cube)
        self.assert_bounds_tick_labels("xaxis")

    def test_yaxis_labels(self):
        iplt.scatter(self.cube, self.cube.coord("str_coord"))
        self.assert_bounds_tick_labels("yaxis")

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 3)
        iplt.scatter(self.cube.coord("str_coord"), self.cube, axes=ax)
        plt.close(fig)
        self.assert_points_tick_labels("xaxis", ax)

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 3)
        iplt.scatter(self.cube, self.cube.coord("str_coord"), axes=ax)
        plt.close(fig)
        self.assert_points_tick_labels("yaxis", ax)

    def test_scatter_longitude(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.scatter(self.lat_lon_cube, self.lat_lon_cube.coord("longitude"), axes=ax)
        plt.close(fig)
