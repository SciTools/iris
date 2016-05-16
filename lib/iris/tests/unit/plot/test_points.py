# (C) British Crown Copyright 2014 - 2016, Met Office
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
"""Unit tests for the `iris.plot.points` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.tests.stock import simple_2d
from iris.tests.unit.plot import TestGraphicStringCoord, MixinCoords

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def test_yaxis_labels(self):
        iplt.points(self.cube, coords=('bar', 'str_coord'))
        self.assertBoundsTickLabels('yaxis')

    def test_xaxis_labels(self):
        iplt.points(self.cube, coords=('str_coord', 'bar'))
        self.assertBoundsTickLabels('xaxis')

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 3)
        iplt.points(self.cube, coords=('str_coord', 'bar'), axes=ax)
        plt.close(fig)
        self.assertPointsTickLabels('xaxis', ax)

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 3)
        iplt.points(self.cube, coords=('bar', 'str_coord'), axes=ax)
        plt.close(fig)
        self.assertPointsTickLabels('yaxis', ax)

    def test_geoaxes_exception(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.assertRaises(TypeError, iplt.points,
                          self.lat_lon_cube, axes=ax)
        plt.close(fig)


@tests.skip_plot
class TestCoords(tests.IrisTest, MixinCoords):
    def setUp(self):
        # We have a 2d cube with dimensionality (bar: 3; foo: 4)
        self.cube = simple_2d(with_bounds=False)
        self.foo = self.cube.coord('foo').points
        self.foo_index = np.arange(self.foo.size)
        self.bar = self.cube.coord('bar').points
        self.bar_index = np.arange(self.bar.size)
        self.data = None
        self.dataT = None
        self.mpl_patch = self.patch('matplotlib.pyplot.scatter')
        self.draw_func = iplt.points


if __name__ == "__main__":
    tests.main()
