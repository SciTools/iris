# (C) British Crown Copyright 2010 - 2012, Met Office
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
Tests the high-level plotting interface.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import matplotlib.pyplot as plt

import iris
import iris.coords as coords
import iris.plot as iplt
import iris.quickplot as qplt
import iris.tests.test_plot as test_plot
import iris.tests.test_mapping as test_mapping


# Caches _load_theta so subsequent calls are faster
def cache(fn, cache={}):
    def inner(*args, **kwargs):
        key = "result"
        if not cache:
            cache[key] =  fn(*args, **kwargs)
        return cache[key]
    return inner


@cache
def _load_theta():
    path = tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog_subset.pp'))
    theta = iris.load_cube(path, 'air_potential_temperature')
    
    # Improve the unit
    theta.units = 'K'

    return theta


@iris.tests.skip_data
class TestQuickplotCoordinatesGiven(test_plot.TestPlotCoordinatesGiven):
    def setUp(self):
        filename = tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog_subset.pp'))
        self.cube = test_plot.load_cube_once(filename, 'air_potential_temperature')
        
        self.draw_module = iris.quickplot
        self.contourf = test_plot.LambdaStr('iris.quickplot.contourf', lambda cube, *args, **kwargs: 
                                                  iris.quickplot.contourf(cube, *args, **kwargs))
        self.contour = test_plot.LambdaStr('iris.quickplot.contour', lambda cube, *args, **kwargs: 
                                                  iris.quickplot.contour(cube, *args, **kwargs))
        self.points = test_plot.LambdaStr('iris.quickplot.points', lambda cube, *args, **kwargs: 
                                                  iris.quickplot.points(cube, c=cube.data, *args, **kwargs))
        self.plot = test_plot.LambdaStr('iris.quickplot.plot', lambda cube, *args, **kwargs: 
                                                  iris.quickplot.plot(cube, *args, **kwargs))
        
        self.results = {'yx': (
                           [self.contourf, ['grid_latitude', 'grid_longitude']],
                           [self.contourf, ['grid_longitude', 'grid_latitude']],
                           [self.contour, ['grid_latitude', 'grid_longitude']],
                           [self.contour, ['grid_longitude', 'grid_latitude']],
                           [self.points, ['grid_latitude', 'grid_longitude']],
                           [self.points, ['grid_longitude', 'grid_latitude']],                   
                           ),
                       'zx': (
                           [self.contourf, ['model_level_number', 'grid_longitude']],
                           [self.contourf, ['grid_longitude', 'model_level_number']],
                           [self.contour, ['model_level_number', 'grid_longitude']],
                           [self.contour, ['grid_longitude', 'model_level_number']],
                           [self.points, ['model_level_number', 'grid_longitude']],
                           [self.points, ['grid_longitude', 'model_level_number']],
                           ),
                        'tx': (
                           [self.contourf, ['time', 'grid_longitude']],
                           [self.contourf, ['grid_longitude', 'time']],
                           [self.contour, ['time', 'grid_longitude']],
                           [self.contour, ['grid_longitude', 'time']],
                           [self.points, ['time', 'grid_longitude']],
                           [self.points, ['grid_longitude', 'time']],
                           ),
                        'x': (
                           [self.plot, ['grid_longitude']],                                      
                           ),
                        'y': (
                           [self.plot, ['grid_latitude']],                                      
                           ),                             
                       }


@iris.tests.skip_data
class TestLabels(tests.GraphicsTest):
    def setUp(self):
        self.theta = _load_theta()

    def _slice(self, coords):
        """Returns the first cube containing the requested coordinates."""
        for cube in self.theta.slices(coords):
            break
        return cube

    def _small(self):
        # Use a restricted size so we can make out the detail
        cube = self._slice(['model_level_number', 'grid_longitude'])
        return cube[:5, :5]

    def test_contour(self):
        qplt.contour(self._small())
        self.check_graphic()
        
        qplt.contourf(self._small(), coords=['model_level_number', 'grid_longitude'])
        self.check_graphic()

    def test_contourf(self):
        qplt.contourf(self._small())
        iplt.orography_at_points(self._small())
        self.check_graphic()
        
        qplt.contourf(self._small(), coords=['model_level_number', 'grid_longitude'])
        self.check_graphic()
        
        qplt.contourf(self._small(), coords=['grid_longitude', 'model_level_number'])
        self.check_graphic()

    def test_contourf_nameless(self):
        cube = self._small()
        cube.standard_name = None
        qplt.contourf(cube, coords=['grid_longitude', 'model_level_number'])
        self.check_graphic()

    def test_pcolor(self):
        qplt.pcolor(self._small())
        self.check_graphic()

    def test_pcolormesh(self):
        qplt.pcolormesh(self._small())

        #cube = self._small()
        #iplt.orography_at_bounds(cube)

        self.check_graphic()

    def test_map(self):
        cube = self._slice(['grid_latitude', 'grid_longitude'])
        qplt.contour(cube)
        self.check_graphic()
        
        # check that the result of adding 360 to the data is *almost* identically the same result
        lon = cube.coord('grid_longitude')
        lon.points = lon.points + 360
        qplt.contour(cube)
        self.check_graphic()

    def test_alignment(self):
        cube = self._small()
        qplt.contourf(cube)
        #qplt.outline(cube)
        qplt.points(cube)
        self.check_graphic()

    def test_num_ticks(self):
        # Test usage via _label_with_points().
        qplt.contourf(self._small(), num_ticks=5)
        self.check_graphic()

        # Test usage through _label_with_bounds().
        qplt.pcolormesh(self._small(), num_ticks=5)
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
