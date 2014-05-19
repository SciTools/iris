# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for the `iris.plot.scatter` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris
from iris.coord_categorisation import add_weekday
import numpy as np

if tests.MPL_AVAILABLE:
    from iris.plot import scatter
    import matplotlib.pyplot as plt


@tests.skip_plot
class TestStringCoordPlot(tests.GraphicsTest):
    def test_scatter_xaxis_labels(self):
        exp_ticklabels = ['', 'a', 'c', 'e', 'b', 'd', '']
        cube = iris.cube.Cube(np.random.rand(10), long_name='foo', units=1)
        str_coord = iris.coords.AuxCoord(np.array(['a', 'b', 'c', 'd', 'e']*2),
                                         long_name='string',
                                         units='1')
        val_coord = iris.coords.AuxCoord(np.random.rand(10),
                                         long_name='val',
                                         units=1)
        cube.add_aux_coord(str_coord, 0)
        cube.add_aux_coord(val_coord, 0)
        scatter(cube.coord('string'), cube.coord('val'), c=cube.data)
        xaxis = plt.gca().xaxis
        ticklabels = [t.get_text() for t in xaxis.get_majorticklabels()]
        self.assertEqual(exp_ticklabels, ticklabels)


if __name__ == "__main__":
    tests.main()
