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
"""Unit tests for the `iris.plot.pcolor` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
from iris.tests.unit.plot import TestGraphicStringCoord

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def test_yaxis_labels(self):
        iplt.pcolor(self.cube, coords=('bar', 'str_coord'))
        self.assertBoundsTickLabels('yaxis')

    def test_xaxis_labels(self):
        iplt.pcolor(self.cube, coords=('str_coord', 'bar'))
        self.assertBoundsTickLabels('xaxis')

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 3)
        iplt.pcolor(self.cube, coords=('str_coord', 'bar'), axes=ax)
        plt.close(fig)
        self.assertPointsTickLabels('xaxis', ax)

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 3)
        iplt.pcolor(self.cube, axes=ax, coords=('bar', 'str_coord'))
        plt.close(fig)
        self.assertPointsTickLabels('yaxis', ax)

    def test_geoaxes_exception(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.assertRaises(TypeError, iplt.pcolor,
                          self.lat_lon_cube, axes=ax)
        plt.close(fig)

if __name__ == "__main__":
    tests.main()
