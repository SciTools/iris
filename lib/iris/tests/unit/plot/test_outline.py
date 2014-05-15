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
"""Unit tests for the `iris.plot.outline` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris
from iris.coord_categorisation import add_weekday

if tests.MPL_AVAILABLE:
    from iris.plot import outline
    import matplotlib.pyplot as plt


@tests.skip_plot
class TestStringCoordPlot(tests.GraphicsTest):
    def test_outline_xaxis_labels(self):
        exp_ticklabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri',
                          'Sat', 'Sun', 'Mon', 'Tue']
        filename = tests.get_data_path(('NetCDF', 'global', 'xyt',
                                        'SMALL_hires_wind_u_for_ipcc4.nc'))
        cube = iris.load_cube(filename)
        add_weekday(cube, cube.coord('time'))
        sub_cube = cube[:, :, 160]
        outline(sub_cube, coords=['weekday', 'latitude'])
        xaxis = plt.gca().xaxis
        ticklabels = [t.get_text() for t in xaxis.get_majorticklabels()]
        self.assertEqual(exp_ticklabels, ticklabels)
        plt.close()


if __name__ == "__main__":
    tests.main()
