# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot.contourf` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from iris.tests.stock import simple_2d
from iris.tests.unit.plot import MixinCoords, TestGraphicStringCoord

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def test_yaxis_labels(self):
        iplt.contourf(self.cube, coords=("bar", "str_coord"))
        self.assertPointsTickLabels("yaxis")

    def test_xaxis_labels(self):
        iplt.contourf(self.cube, coords=("str_coord", "bar"))
        self.assertPointsTickLabels("xaxis")

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.contourf(self.cube, axes=ax, coords=("bar", "str_coord"))
        plt.close(fig)
        self.assertPointsTickLabels("yaxis", ax)

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.contourf(self.cube, axes=ax, coords=("str_coord", "bar"))
        plt.close(fig)
        self.assertPointsTickLabels("xaxis", ax)

    def test_geoaxes_exception(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.assertRaises(TypeError, iplt.contourf, self.lat_lon_cube, axes=ax)
        plt.close(fig)


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
        self.draw_func = iplt.contourf


@tests.skip_plot
class TestAntialias(tests.IrisTest):
    def test_skip_contour(self):
        # Contours should not be added if data is all below second level.  See #4086.
        cube = simple_2d()

        levels = [5, 15, 20, 200]
        colors = ["b", "r", "y"]

        iplt.contourf(cube, levels=levels, colors=colors, antialiased=True)

        ax = plt.gca()
        # Expect 3 PathCollection objects (one for each colour) and no LineCollection
        # objects.
        for collection in ax.collections:
            self.assertIsInstance(
                collection, matplotlib.collections.PathCollection
            )
        self.assertEqual(len(ax.collections), 3)


if __name__ == "__main__":
    tests.main()
