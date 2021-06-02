# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot.scatter` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
from iris.tests.unit.plot import TestGraphicStringCoord

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def setUp(self):
        super().setUp()
        self.cube = self.cube[0, :]
        self.lat_lon_cube = self.lat_lon_cube[0, :]

    def test_xaxis_labels(self):
        iplt.scatter(self.cube.coord("str_coord"), self.cube)
        self.assertBoundsTickLabels("xaxis")

    def test_yaxis_labels(self):
        iplt.scatter(self.cube, self.cube.coord("str_coord"))
        self.assertBoundsTickLabels("yaxis")

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 3)
        iplt.scatter(self.cube.coord("str_coord"), self.cube, axes=ax)
        plt.close(fig)
        self.assertPointsTickLabels("xaxis", ax)

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 3)
        iplt.scatter(self.cube, self.cube.coord("str_coord"), axes=ax)
        plt.close(fig)
        self.assertPointsTickLabels("yaxis", ax)

    def test_scatter_longitude(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        iplt.scatter(
            self.lat_lon_cube, self.lat_lon_cube.coord("longitude"), axes=ax
        )
        plt.close(fig)


if __name__ == "__main__":
    tests.main()
