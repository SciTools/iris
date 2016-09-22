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
"""Integration tests for the package :mod:`iris.quickplot`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests
import iris
from iris.tests.integration.plotting import \
     TestNDCoordinatesGiven, LambdaStr, load_cube_once, load_theta

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt
    from iris.plot import (plot, contour, contourf, pcolor, pcolormesh, points)


@tests.skip_data
@tests.skip_plot
class TestAttributePositive(tests.GraphicsTest):
    def test_1d_positive_up(self):
        path = tests.get_data_path(('NetCDF', 'ORCA2', 'votemper.nc'))
        cube = iris.load_cube(path)
        plot(cube.coord('depth'), cube[0, :, 60, 80])
        self.check_graphic()

    def test_1d_positive_down(self):
        path = tests.get_data_path(('NetCDF', 'ORCA2', 'votemper.nc'))
        cube = iris.load_cube(path)
        plot(cube[0, :, 60, 80], cube.coord('depth'))
        self.check_graphic()

    def test_2d_positive_up(self):
        path = tests.get_data_path(('NetCDF', 'testing',
                                    'small_theta_colpex.nc'))
        cube = iris.load_cube(path)[0, :, 42, :]
        pcolormesh(cube)
        self.check_graphic()

    def test_2d_positive_down(self):
        path = tests.get_data_path(('NetCDF', 'ORCA2', 'votemper.nc'))
        cube = iris.load_cube(path)[0, :, 42, :]
        pcolormesh(cube)
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestQuickplotCoordinatesGiven(TestNDCoordinatesGiven):
    def setUp(self):
        filename = tests.get_data_path(
            ('PP', 'COLPEX', 'theta_and_orog_subset.pp'))
        self.cube = load_cube_once(filename, 'air_potential_temperature')

        self.contourf = LambdaStr('iris.quickplot.contourf',
                                  lambda cube, *args, **kwargs:
                                  contourf(cube, *args, **kwargs))
        self.contour = LambdaStr('iris.quickplot.contour',
                                 lambda cube, *args, **kwargs:
                                 contour(cube, *args, **kwargs))
        self.pcolor = LambdaStr('iris.quickplot.pcolor',
                                lambda cube, *args, **kwargs:
                                pcolor(cube, *args, **kwargs))
        self.pcolormesh = LambdaStr('iris.quickplot.pcolormesh',
                                    lambda cube, *args, **kwargs:
                                    pcolormesh(cube, *args, **kwargs))
        self.points = LambdaStr('iris.quickplot.points',
                                lambda cube, *args, **kwargs:
                                points(cube, c=cube.data, *args, **kwargs))
        self.plot = LambdaStr('iris.quickplot.plot',
                              lambda cube, *args, **kwargs:
                              plot(cube, *args, **kwargs))

        super(TestQuickplotCoordinatesGiven, self).__init__()


@tests.skip_data
@tests.skip_plot
class TestLabels(tests.GraphicsTest):
    def setUp(self):
        self.theta = load_theta()

    def _slice(self, coords):
        """Returns the first cube containing the requested coordinates."""
        for cube in self.theta.slices(coords):
            break
        return cube

    def _small(self):
        # Use a restricted size so we can make out the detail.
        cube = self._slice(['model_level_number', 'grid_longitude'])
        return cube[:5, :5]

    def test_contour(self):
        contour(self._small())
        self.check_graphic()

        contour(self._small(),
                coords=['model_level_number', 'grid_longitude'])
        self.check_graphic()

    def test_contourf(self):
        contourf(self._small())
        self.check_graphic()

        contourf(self._small(),
                 coords=['model_level_number', 'grid_longitude'])
        self.check_graphic()

        contourf(self._small(),
                 coords=['grid_longitude', 'model_level_number'])
        self.check_graphic()

    def test_contourf_nameless(self):
        cube = self._small()
        cube.standard_name = None
        contourf(cube,
                 coords=['grid_longitude', 'model_level_number'])
        self.check_graphic()

    def test_pcolor(self):
        pcolor(self._small())
        self.check_graphic()

    def test_pcolormesh(self):
        pcolormesh(self._small())
        self.check_graphic()

    def test_map(self):
        cube = self._slice(['grid_latitude', 'grid_longitude'])
        contour(cube)
        self.check_graphic()

    def test_add_roll(self):
        # Check that the result of adding 360 to the data is almost identical.
        cube = self._slice(['grid_latitude', 'grid_longitude'])
        lon = cube.coord('grid_longitude')
        lon.points = lon.points + 360
        contour(cube)
        self.check_graphic()

    def test_alignment(self):
        cube = self._small()
        contourf(cube)
        points(cube)
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestTimeReferenceUnitsLabels(tests.GraphicsTest):

    def setUp(self):
        path = tests.get_data_path(('PP', 'aPProt1', 'rotatedMHtimecube.pp'))
        self.cube = iris.load_cube(path)[:, 0, 0]

    def test_reference_time_units(self):
        # units should not be displayed for a reference time
        plot(self.cube.coord('time'), self.cube)
        plt.gcf().autofmt_xdate()
        self.check_graphic()

    def test_not_reference_time_units(self):
        # units should be displayed for other time coordinates
        plot(self.cube.coord('forecast_period'), self.cube)
        self.check_graphic()


if __name__ == "__main__":
    tests.main()