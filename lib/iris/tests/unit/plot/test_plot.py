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
"""Unit tests for the `iris.plot.plot` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris
from iris.coord_categorisation import add_weekday
from iris.tests.stock import realistic_4d

if tests.MPL_AVAILABLE:
    from iris.plot import plot
    import matplotlib.pyplot as plt


@tests.skip_plot
class TestStringCoordPlot(tests.GraphicsTest):
    def test_plot_xaxis_labels(self):
        exp_ticklabels = ['Wed', 'Wed', 'Wed', 'Wed', 'Wed', 'Wed']
        cube = realistic_4d()
        add_weekday(cube, cube.coord('time'))
        sub_cube = cube[:, 0, 40, 60]
        plot(sub_cube.coord('weekday'), sub_cube)
        xaxis = plt.gca().xaxis
        ticklabels = [t.get_text() for t in xaxis.get_majorticklabels()]
        self.assertEqual(exp_ticklabels, ticklabels)


if __name__ == "__main__":
    tests.main()
