# (C) British Crown Copyright 2016, Met Office
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
"""Integration tests for the package :mod:`iris.plot`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests
import iris

import iris.tests.integration.plotting as plotting

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt
    from iris.plot import (plot, contour, contourf, pcolor, pcolormesh,
                           scatter, points, orography_at_points)


@tests.skip_plot
class TestSimple(tests.GraphicsTest):
    def test_points(self):
        cube = plotting.simple_cube()
        contourf(cube)
        self.check_graphic()

    def test_bounds(self):
        cube = plotting.simple_cube()
        pcolor(cube)
        self.check_graphic()


@tests.skip_plot
class TestMissingCoord(tests.GraphicsTest):
    def _check(self, cube):
        contourf(cube)
        self.check_graphic()
        pcolor(cube)
        self.check_graphic()

    def test_no_u(self):
        cube = plotting.simple_cube()
        cube.remove_coord('grid_longitude')
        self._check(cube)

    def test_no_v(self):
        cube = plotting.simple_cube()
        cube.remove_coord('time')
        self._check(cube)

    def test_none(self):
        cube = plotting.simple_cube()
        cube.remove_coord('grid_longitude')
        cube.remove_coord('time')
        self._check(cube)


@tests.skip_plot
class TestHybridHeight(tests.GraphicsTest):
    def setUp(self):
        self.cube = iris.tests.stock.realistic_4d()[0, :15, 0, :]

    def test_points(self):
        contourf(self.cube, coords=['level_height', 'grid_longitude'])
        self.check_graphic()
        contourf(self.cube, coords=['altitude', 'grid_longitude'])
        self.check_graphic()

    def test_orography(self):
        contourf(self.cube)
        orography_at_points(self.cube)
        self.check_graphic()

    def test_orography_specific_coords(self):
        coords = ['altitude', 'grid_longitude']
        contourf(self.cube, coords=coords)
        orography_at_points(self.cube, coords=coords)
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestMissingCoordSystem(tests.GraphicsTest):
    def test(self):
        cube = tests.stock.simple_pp()
        cube.coord("latitude").coord_system = None
        cube.coord("longitude").coord_system = None
        contourf(cube)
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class Test1dPlotScatter(plotting.Test1dScatter):

    def setUp(self):
        self.cube = iris.load_cube(
            tests.get_data_path(('NAME', 'NAMEIII_trajectory.txt')),
            'Temperature')
        self.draw_method = scatter
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestPlotCoordinatesGiven(plotting.TestNDCoordinatesGiven):
    def setUp(self):
        filename = tests.get_data_path(('PP', 'COLPEX',
                                        'theta_and_orog_subset.pp'))
        self.cube = plotting.load_cube_once(filename,
                                            'air_potential_temperature')

        self.draw_module = iris.plot
        self.contourf = plotting.LambdaStr('iris.plot.contourf',
                                           lambda cube, *args, **kwargs:
                                           contourf(cube, *args, **kwargs))
        self.contour = plotting.LambdaStr('iris.plot.contour',
                                          lambda cube, *args, **kwargs:
                                          contour(cube, *args, **kwargs))
        self.pcolor = plotting.LambdaStr('iris.quickplot.pcolor',
                                         lambda cube, *args, **kwargs:
                                         pcolor(cube, *args, **kwargs))
        self.pcolormesh = plotting.LambdaStr('iris.quickplot.pcolormesh',
                                             lambda cube, *args, **kwargs:
                                             pcolormesh(cube, *args, **kwargs))
        self.points = plotting.LambdaStr('iris.plot.points',
                                         lambda cube, *args, **kwargs:
                                         points(cube, c=cube.data,
                                                *args, **kwargs))
        self.plot = plotting.LambdaStr('iris.plot.plot',
                                       lambda cube, *args, **kwargs:
                                       plot(cube, *args, **kwargs))

        super(TestPlotCoordinatesGiven, self).setUp()


if __name__ == "__main__":
    tests.main()
