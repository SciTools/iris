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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import matplotlib.pyplot as plt 
import numpy

import iris
import iris.coords as coords
import iris.plot as iplt
import iris.quickplot as qplt
import iris.symbols
import iris.tests.stock
import iris.tests.test_mapping as test_mapping


def simple_cube():
    cube = iris.tests.stock.realistic_4d()
    cube = cube[:, 0, 0, :]
    cube.coord('time').guess_bounds()
    return cube


class TestSimple(tests.GraphicsTest):
    def test_points(self):
        cube = simple_cube()
        qplt.contourf(cube)
        self.check_graphic()

    def test_bounds(self):
        cube = simple_cube()
        qplt.pcolor(cube)
        self.check_graphic()


class TestMissingCoord(tests.GraphicsTest):
    def _check(self, cube):
        qplt.contourf(cube)
        self.check_graphic()

        qplt.pcolor(cube)
        self.check_graphic()

    def test_no_u(self):
        cube = simple_cube()
        cube.remove_coord('grid_longitude')
        self._check(cube)

    def test_no_v(self):
        cube = simple_cube()
        cube.remove_coord('time')
        self._check(cube)

    def test_none(self):
        cube = simple_cube()
        cube.remove_coord('grid_longitude')
        cube.remove_coord('time')
        self._check(cube)


class TestHybridHeight(tests.GraphicsTest):
    def setUp(self):
        self.cube = iris.tests.stock.realistic_4d()[0, :15, 0, :]

    def _check(self, plt_method, test_altitude=True):
        plt_method(self.cube)
        self.check_graphic()

        plt_method(self.cube, coords=['level_height', 'grid_longitude'])
        self.check_graphic()

        plt_method(self.cube, coords=['grid_longitude', 'level_height'])
        self.check_graphic()

        if test_altitude:
            plt_method(self.cube, coords=['grid_longitude', 'altitude'])
            self.check_graphic()
        
            plt_method(self.cube, coords=['altitude', 'grid_longitude'])
            self.check_graphic()

    def test_points(self):
        self._check(qplt.contourf)

    def test_bounds(self):
        self._check(qplt.pcolor, test_altitude=False)

    def test_orography(self):
        qplt.contourf(self.cube)
        iplt.orography_at_points(self.cube)
        iplt.points(self.cube)
        self.check_graphic()

        coords = ['altitude', 'grid_longitude']
        qplt.contourf(self.cube, coords=coords)
        iplt.orography_at_points(self.cube, coords=coords)
        iplt.points(self.cube, coords=coords)
        self.check_graphic()
        
        # TODO: Test bounds once they are supported.
        with self.assertRaises(NotImplementedError):
            qplt.pcolor(self.cube)
            iplt.orography_at_bounds(self.cube)
            iplt.outline(self.cube)
            self.check_graphic()


# Caches _load_wind so subsequent calls are faster
def cache(fn, cache={}):
    def inner(*args, **kwargs):
        key = "result"
        if not cache:
            cache[key] =  fn(*args, **kwargs)
        return cache[key]
    return inner


@cache
def _load_wind():
    # Load the COLPEX data => TZYX
    path = tests.get_data_path(('PP', 'COLPEX', 'uwind_and_orog.pp'))
    wind = iris.load_strict(path, 'eastward_wind')

    # Until there is better mapping support for rotated-pole, pretend this isn't rotated.
    # ie. Move the pole from (37.5, 177.5) to (90, 0) and shift the coordinates.
    tests.test_mapping._pretend_unrotated(wind)
    
    # Add time bounds so we can test for bounded time plots
    flt = wind.coord('forecast_period')
    lower = numpy.arange(6, 12, dtype=numpy.float32) / 6
    upper = numpy.arange(7, 13, dtype=numpy.float32) / 6
    flt._bounds = numpy.column_stack([lower, upper])

    return wind[:, :, :50, :50]


def _time_series(src_cube):
    # Until we have plotting support for multiple axes on the same dimension,
    # remove the time coordinate and its axis.
    cube = src_cube.copy()
    cube.remove_coord('time')
    return cube


def _date_series(src_cube):
    # Until we have plotting support for multiple axes on the same dimension,
    # remove the forecast_period coordinate and its axis.
    cube = src_cube.copy()
    cube.remove_coord('forecast_period')
    return cube


class SliceMixin(object):
    """Mixin class providing tests for each 2-dimensional permutation of axes.

    Requires self.draw_method to be the relevant plotting function,
    and self.results to be a dictionary containing the desired test results."""

    def test_yx(self):
        cube = self.wind[0, 0, :, :]
        iplt.map_setup(cube=cube, mode=coords.POINT_MODE)
        self.draw_method(cube)
        self.check_graphic()

    def test_zx(self):
        cube = self.wind[0, :, 0, :]
        self.draw_method(cube)
        self.check_graphic()

    def test_tx(self):
        cube = _time_series(self.wind[:, 0, 0, :])
        self.draw_method(cube)
        self.check_graphic()

    def test_zy(self):
        cube = self.wind[0, :, :, 0]
        self.draw_method(cube)
        self.check_graphic()

    def test_ty(self):
        cube = _time_series(self.wind[:, 0, :, 0])
        self.draw_method(cube)
        self.check_graphic()

    def test_tz(self):
        cube = _time_series(self.wind[:, :, 0, 0])
        self.draw_method(cube)
        self.check_graphic()


@iris.tests.skip_data
class TestContour(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.contour routine."""
    def setUp(self):
        self.wind = _load_wind()
        self.draw_method = iplt.contour


@iris.tests.skip_data
class TestContourf(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.contourf routine."""
    def setUp(self):
        self.wind = _load_wind()
        self.draw_method = iplt.contourf


@iris.tests.skip_data
class TestPcolor(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.pcolor routine."""
    def setUp(self):
        self.wind = _load_wind()
        self.draw_method = iplt.pcolor


@iris.tests.skip_data
class TestPcolormesh(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.pcolormesh routine."""
    def setUp(self):
        self.wind = _load_wind()
        self.draw_method = iplt.pcolormesh


class Slice1dMixin(object):
    """Mixin class providing tests for each 1-dimensional permutation of axes.

    Requires self.draw_method to be the relevant plotting function,
    and self.results to be a dictionary containing the desired test results."""
    
    def test_x(self):
        cube = self.wind[0, 0, 0, :]
        self.draw_method(cube)
        self.check_graphic()
        
    def test_y(self):
        cube = self.wind[0, 0, :, 0]
        self.draw_method(cube)
        self.check_graphic()
        
    def test_z(self):
        cube = self.wind[0, :, 0, 0]
        self.draw_method(cube)
        self.check_graphic()

    def test_t(self):
        cube = _time_series(self.wind[:, 0, 0, 0])
        self.draw_method(cube)
        self.check_graphic()
        
    def test_t_dates(self):
        cube = _date_series(self.wind[:, 0, 0, 0])
        self.draw_method(cube)
        plt.gcf().autofmt_xdate()
        plt.xlabel('Phenomenon time')

        self.check_graphic()


@iris.tests.skip_data
class TestPlot(tests.GraphicsTest, Slice1dMixin):
    """Test the iris.plot.plot routine."""
    def setUp(self):
        self.wind = _load_wind()
        self.draw_method = iplt.plot
        

@iris.tests.skip_data        
class TestQuickplotPlot(tests.GraphicsTest, Slice1dMixin):
    """Test the iris.quickplot.plot routine."""
    def setUp(self):
        self.wind = _load_wind()
        self.draw_method = qplt.plot


@iris.tests.skip_data
class TestFillContinents(tests.GraphicsTest):
    def setUp(self):
        datafile = tests.get_data_path(('PP', 'itam', 'WO0000000000934', 'NAE.20100908_00_an.pp'))
        self.cube = iris.load(datafile)[0]
        # scale down the data by 100
        self.cube.data = self.cube.data/100.0
    
    def test_fillcontinents_underneath(self):
        
        # setup the map and plot output
        current_map = iris.plot.map_setup(resolution='i', lon_range=[-70, 70], lat_range=[25, 75], projection='merc')
        current_map.drawcoastlines()
        current_map.fillcontinents(color='green', lake_color='aqua', zorder=0)
        iris.plot.contourf(self.cube)
        
        self.check_graphic()

    def test_fillcontinents_ontop(self):

        # setup the map and plot output
        current_map = iris.plot.map_setup(resolution='i', lon_range=[-70, 70], lat_range=[25, 75], projection='merc')
        current_map.drawcoastlines()
        current_map.fillcontinents(color='green', lake_color='aqua', zorder=3)
        iris.plot.contourf(self.cube)
        
        self.check_graphic()


_load_strict_once_cache = {}


def load_strict_once(filename, constraint):
    """Same syntax as load_strict, but will only load a file once, then cache the answer in a dictionary."""
    global _load_strict_once_cache
    key = (filename, str(constraint))
    cube = _load_strict_once_cache.get(key, None)
    
    if cube is None:
        cube = iris.load_strict(filename, constraint)
        _load_strict_once_cache[key] = cube
        
    return cube


class LambdaStr(object):
    """Provides a callable function which has a sensible __repr__.""" 
    def __init__(self, repr, lambda_fn):
        self.repr = repr
        self.lambda_fn = lambda_fn
        
    def __call__(self, *args, **kwargs):
        return self.lambda_fn(*args, **kwargs)

    def __repr__(self):
        return self.repr


@iris.tests.skip_data
class TestPlotCoordinatesGiven(tests.GraphicsTest):
    def setUp(self):
        filename = tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog_subset.pp'))
        self.cube = load_strict_once(filename, 'air_potential_temperature')
        
        self.draw_module = iris.plot
        self.contourf = LambdaStr('iris.plot.contourf', lambda cube, *args, **kwargs: iris.plot.contourf(cube, *args, **kwargs))
        self.contour = LambdaStr('iris.plot.contour', lambda cube, *args, **kwargs: iris.plot.contour(cube, *args, **kwargs))
        self.points = LambdaStr('iris.plot.points', lambda cube, *args, **kwargs: iris.plot.points(cube, c=cube.data,
                                                                     *args, **kwargs))
        self.plot = LambdaStr('iris.plot.plot', lambda cube, *args, **kwargs: iris.plot.plot(cube, *args, **kwargs))
        
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
        
    def draw(self, draw_method, *args, **kwargs):
        draw_fn = getattr(self.draw_module, draw_method)
        draw_fn(*args, **kwargs)
        self.check_graphic()
    
    def run_tests(self, cube, results):
        for draw_method, coords in results:
            draw_method(cube, coords=coords)
            try:
                self.check_graphic()
            except AssertionError, err:
                self.fail('Draw method %r failed with coords: %r. Assertion message: %s' % (draw_method, coords, err))
            
    def test_yx(self):
        test_cube = self.cube[0, 0, :, :]        
        self.run_tests(test_cube, self.results['yx'])
        
    def test_zx(self):
        test_cube = self.cube[0, :15, 0, :]
        self.run_tests(test_cube, self.results['zx'])
        
    def test_tx(self):
        test_cube = self.cube[:, 0, 0, :]
        self.run_tests(test_cube, self.results['tx'])
        
    def test_x(self):
        test_cube = self.cube[0, 0, 0, :]
        self.run_tests(test_cube, self.results['x'])
        
    def test_y(self):
        test_cube = self.cube[0, 0, :, 0]
        self.run_tests(test_cube, self.results['y'])
        
    def test_badcoords(self):
        cube = self.cube[0, 0, :, :]
        draw_fn = getattr(self.draw_module, 'contourf')
        self.assertRaises(ValueError, draw_fn, cube, coords=['grid_longitude', 'grid_longitude'])
        self.assertRaises(ValueError, draw_fn, cube, coords=['grid_longitude', 'grid_longitude', 'grid_latitude'])
        self.assertRaises(iris.exceptions.CoordinateNotFoundError, draw_fn, cube, coords=['grid_longitude', 'wibble'])
        self.assertRaises(ValueError, draw_fn, cube, coords=[])
        self.assertRaises(ValueError, draw_fn, cube, coords=[cube.coord('grid_longitude'), cube.coord('grid_longitude')])
        self.assertRaises(ValueError, draw_fn, cube, coords=[cube.coord('grid_longitude'),
                                                             cube.coord('grid_longitude'),
                                                             cube.coord('grid_longitude')])
        
    def test_non_cube_coordinate(self):
        cube = self.cube[0, :, :, 0]
        pts = -100 + numpy.arange(cube.shape[1]) * 13
        x = coords.DimCoord(pts, 'model_level_number', attributes={'positive': 'up'})
        self.draw('contourf', cube, coords=['grid_latitude', x])


class TestSymbols(tests.GraphicsTest):
    def test_cloud_cover(self):
        iplt.symbols(range(10), [0] * 10, [iris.symbols.CLOUD_COVER[i] for i in range(10)], 0.375)
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
