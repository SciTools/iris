# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.quickplot.contourf` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.tests.stock import simple_2d
from iris.tests.unit.plot import MixinCoords, TestGraphicStringCoord

if tests.MPL_AVAILABLE:
    import iris.quickplot as qplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def test_yaxis_labels(self):
        qplt.contourf(self.cube, coords=("bar", "str_coord"))
        self.assertPointsTickLabels("yaxis")

    def test_xaxis_labels(self):
        qplt.contourf(self.cube, coords=("str_coord", "bar"))
        self.assertPointsTickLabels("xaxis")


@tests.skip_plot
class TestCoords(tests.IrisTest, MixinCoords):
    def setUp(self):
        # We have a 2d cube with dimensionality (bar: 3; foo: 4)
        self.cube = simple_2d(with_bounds=False)
        self.foo = self.cube.coord("foo").points
        self.foo_index = np.arange(self.foo.size)
        self.bar = self.cube.coord("bar").points
        self.bar_index = np.arange(self.bar.size)
        self.data = self.cube.data
        self.dataT = self.data.T
        mocker = mock.Mock(alpha=0, antialiased=False)
        self.mpl_patch = self.patch(
            "matplotlib.pyplot.contourf", return_value=mocker
        )
        # Also need to mock the colorbar.
        self.patch("matplotlib.pyplot.colorbar")
        self.draw_func = qplt.contourf


if __name__ == "__main__":
    tests.main()
