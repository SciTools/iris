# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.quickplot.plot` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip
from iris.tests.stock import simple_1d
from iris.tests.unit.plot import TestGraphicStringCoord

if tests.MPL_AVAILABLE:
    import iris.quickplot as qplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def setUp(self):
        super().setUp()
        self.cube = self.cube[0, :]

    def test_yaxis_labels(self):
        qplt.plot(self.cube, self.cube.coord("str_coord"))
        self.assertBoundsTickLabels("yaxis")

    def test_xaxis_labels(self):
        qplt.plot(self.cube.coord("str_coord"), self.cube)
        self.assertBoundsTickLabels("xaxis")


class TestAxisLabels(tests.GraphicsTest):
    def test_xy_cube(self):
        c = simple_1d()
        qplt.plot(c)
        ax = qplt.plt.gca()
        x = ax.xaxis.get_label().get_text()
        self.assertEqual(x, "Foo")
        y = ax.yaxis.get_label().get_text()
        self.assertEqual(y, "Thingness")

    def test_yx_cube(self):
        c = simple_1d()
        c.transpose()
        # Making the cube a vertical coordinate should change the default
        # orientation of the plot.
        c.coord("foo").attributes["positive"] = "up"
        qplt.plot(c)
        ax = qplt.plt.gca()
        x = ax.xaxis.get_label().get_text()
        self.assertEqual(x, "Thingness")
        y = ax.yaxis.get_label().get_text()
        self.assertEqual(y, "Foo")


if __name__ == "__main__":
    tests.main()
