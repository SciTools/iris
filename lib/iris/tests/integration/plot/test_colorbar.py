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
"""
Test interaction between :mod:`iris.plot` and
:func:`matplotlib.pyplot.colorbar`

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.coords import AuxCoord
import iris.tests.stock

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt
    from iris.plot import contour, contourf, pcolormesh, pcolor,\
        points, scatter


@tests.skip_plot
class TestColorBarCreation(tests.GraphicsTest):
    def setUp(self):
        self.draw_functions = (contour, contourf, pcolormesh, pcolor)
        self.cube = iris.tests.stock.lat_lon_cube()
        self.cube.coord('longitude').guess_bounds()
        self.cube.coord('latitude').guess_bounds()
        self.traj_lon = AuxCoord(np.linspace(-180, 180, 50),
                                 standard_name='longitude',
                                 units='degrees')
        self.traj_lat = AuxCoord(np.sin(np.deg2rad(self.traj_lon.points))*30.0,
                                 standard_name='latitude',
                                 units='degrees')

    def test_common_draw_functions(self):
        for draw_function in self.draw_functions:
            mappable = draw_function(self.cube)
            cbar = plt.colorbar()
            self.assertIs(
                cbar.mappable, mappable,
                msg='Problem with draw function iris.plot.{}'.format(
                    draw_function.__name__))

    def test_common_draw_functions_specified_mappable(self):
        for draw_function in self.draw_functions:
            mappable_initial = draw_function(self.cube, cmap='cool')
            mappable = draw_function(self.cube)
            cbar = plt.colorbar(mappable_initial)
            self.assertIs(
                cbar.mappable, mappable_initial,
                msg='Problem with draw function iris.plot.{}'.format(
                    draw_function.__name__))

    def test_points_with_c_kwarg(self):
        mappable = points(self.cube, c=self.cube.data)
        cbar = plt.colorbar()
        self.assertIs(cbar.mappable, mappable)

    def test_points_with_c_kwarg_specified_mappable(self):
        mappable_initial = points(self.cube, c=self.cube.data, cmap='cool')
        mappable = points(self.cube, c=self.cube.data)
        cbar = plt.colorbar(mappable_initial)
        self.assertIs(cbar.mappable, mappable_initial)

    def test_scatter_with_c_kwarg(self):
        mappable = scatter(self.traj_lon, self.traj_lat,
                           c=self.traj_lon.points)
        cbar = plt.colorbar()
        self.assertIs(cbar.mappable, mappable)

    def test_scatter_with_c_kwarg_specified_mappable(self):
        mappable_initial = scatter(self.traj_lon, self.traj_lat,
                                   c=self.traj_lon.points)
        mappable = scatter(self.traj_lon, self.traj_lat,
                           c=self.traj_lon.points,
                           cmap='cool')
        cbar = plt.colorbar(mappable_initial)
        self.assertIs(cbar.mappable, mappable_initial)


if __name__ == "__main__":
    tests.main()
