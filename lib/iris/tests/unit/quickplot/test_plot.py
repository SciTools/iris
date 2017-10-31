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
"""Unit tests for the `iris.quickplot.plot` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
from iris.tests.stock import simple_2d
from iris.tests.unit.plot import TestGraphicStringCoord

if tests.MPL_AVAILABLE:
    import iris.quickplot as qplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def setUp(self):
        super(TestStringCoordPlot, self).setUp()
        self.cube = self.cube[0, :]

    def test_yaxis_labels(self):
        qplt.plot(self.cube, self.cube.coord('str_coord'))
        self.assertPointsTickLabels('yaxis')

    def test_xaxis_labels(self):
        qplt.plot(self.cube.coord('str_coord'), self.cube)
        self.assertPointsTickLabels('xaxis')


class TestAxisLabels(tests.GraphicsTest):
    def test_xy_cube(self):
        c = simple_2d()[:, 0]
        qplt.plot(c)
        ax = qplt.plt.gca()
        x = ax.xaxis.get_label().get_text()
        self.assertEqual(x, 'Bar')
        y = ax.yaxis.get_label().get_text()
        self.assertEqual(y, 'Thingness')

    def test_yx_cube(self):
        c = simple_2d()[:, 0]
        c.transpose()
        # Making the cube a vertical coordinate should change the default
        # orientation of the plot.
        c.coord('bar').attributes['positive'] = 'up'
        qplt.plot(c)
        ax = qplt.plt.gca()
        x = ax.xaxis.get_label().get_text()
        self.assertEqual(x, 'Thingness')
        y = ax.yaxis.get_label().get_text()
        self.assertEqual(y, 'Bar')

if __name__ == "__main__":
    tests.main()
