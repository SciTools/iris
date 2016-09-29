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
"""
Integration tests for the packages :mod:`iris.plot` and :mod:`iris.quickplot`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import iris.tests as tests

from functools import wraps
import types
import warnings

import cf_units
import matplotlib.pyplot as plt
import numpy as np

import iris.coords as coords
from iris.exceptions import CoordinateNotFoundError
import iris.tests.stock


# Helper functions and classes.

_LOAD_CUBE_ONCE_CACHE = {}


def load_cube_once(filename, constraint):
    """Same syntax as load_cube, but will only load a file once,

    then cache the answer in a dictionary.

    """
    global _LOAD_CUBE_ONCE_CACHE
    key = (filename, str(constraint))
    cube = _LOAD_CUBE_ONCE_CACHE.get(key, None)

    if cube is None:
        cube = iris.load_cube(filename, constraint)
        _LOAD_CUBE_ONCE_CACHE[key] = cube

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


def simple_cube():
    cube = iris.tests.stock.realistic_4d()
    cube = cube[:, 0, 0, :]
    cube.coord('time').guess_bounds()
    return cube


def load_theta():
    path = tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog_subset.pp'))
    theta = load_cube_once(path, 'air_potential_temperature')

    # Improve the unit
    theta.units = 'K'

    return theta


# Permutations classes.


@tests.skip_data
@tests.skip_plot
class Test1dScatter(tests.GraphicsTest):

    def test_coord_coord(self):
        x = self.cube.coord('longitude')
        y = self.cube.coord('altitude')
        c = self.cube.data
        self.draw_method(x, y, c=c, edgecolor='none')
        self.check_graphic()

    def test_coord_coord_map(self):
        x = self.cube.coord('longitude')
        y = self.cube.coord('latitude')
        c = self.cube.data
        self.draw_method(x, y, c=c, edgecolor='none')
        plt.gca().coastlines()
        self.check_graphic()

    def test_coord_cube(self):
        x = self.cube.coord('latitude')
        y = self.cube
        c = self.cube.coord('Travel Time').points
        self.draw_method(x, y, c=c, edgecolor='none')
        self.check_graphic()

    def test_cube_coord(self):
        x = self.cube
        y = self.cube.coord('altitude')
        c = self.cube.coord('Travel Time').points
        self.draw_method(x, y, c=c, edgecolor='none')
        self.check_graphic()

    def test_cube_cube(self):
        x = iris.load_cube(
            tests.get_data_path(('NAME', 'NAMEIII_trajectory.txt')),
            'Rel Humidity')
        y = self.cube
        c = self.cube.coord('Travel Time').points
        self.draw_method(x, y, c=c, edgecolor='none')
        self.check_graphic()

    def test_incompatible_objects(self):
        # cubes/coordinates of different sizes cannot be plotted
        x = self.cube
        y = self.cube.coord('altitude')[:-1]
        with self.assertRaises(ValueError):
            self.draw_method(x, y)

    def test_not_cube_or_coord(self):
        # inputs must be cubes or coordinates
        x = np.arange(self.cube.shape[0])
        y = self.cube
        with self.assertRaises(TypeError):
            self.draw_method(x, y)


@tests.skip_data
@tests.skip_plot
class TestNDCoordinatesGiven(tests.GraphicsTest):
    def setUp(self):
        self.results = {'yx': ([self.contourf, ['grid_latitude',
                                                'grid_longitude']],
                               [self.contourf, ['grid_longitude',
                                                'grid_latitude']],
                               [self.contour, ['grid_latitude',
                                               'grid_longitude']],
                               [self.contour, ['grid_longitude',
                                               'grid_latitude']],
                               [self.pcolor, ['grid_latitude',
                                              'grid_longitude']],
                               [self.pcolor, ['grid_longitude',
                                              'grid_latitude']],
                               [self.pcolormesh, ['grid_latitude',
                                                  'grid_longitude']],
                               [self.pcolormesh, ['grid_longitude',
                                                  'grid_latitude']],
                               [self.points, ['grid_latitude',
                                              'grid_longitude']],
                               [self.points, ['grid_longitude',
                                              'grid_latitude']],),
                        'zx': ([self.contourf, ['model_level_number',
                                                'grid_longitude']],
                               [self.contourf, ['grid_longitude',
                                                'model_level_number']],
                               [self.contour, ['model_level_number',
                                               'grid_longitude']],
                               [self.contour, ['grid_longitude',
                                               'model_level_number']],
                               [self.pcolor, ['model_level_number',
                                              'grid_longitude']],
                               [self.pcolor, ['grid_longitude',
                                              'model_level_number']],
                               [self.pcolormesh, ['model_level_number',
                                                  'grid_longitude']],
                               [self.pcolormesh, ['grid_longitude',
                                                  'model_level_number']],
                               [self.points, ['model_level_number',
                                              'grid_longitude']],
                               [self.points, ['grid_longitude',
                                              'model_level_number']],),
                        'tx': ([self.contourf, ['time', 'grid_longitude']],
                               [self.contourf, ['grid_longitude', 'time']],
                               [self.contour, ['time', 'grid_longitude']],
                               [self.contour, ['grid_longitude', 'time']],
                               [self.pcolor, ['time', 'grid_longitude']],
                               [self.pcolor, ['grid_longitude', 'time']],
                               [self.pcolormesh, ['time', 'grid_longitude']],
                               [self.pcolormesh, ['grid_longitude', 'time']],
                               [self.points, ['time', 'grid_longitude']],
                               [self.points, ['grid_longitude', 'time']],),
                        'x': ([self.plot, ['grid_longitude']],),
                        'y': ([self.plot, ['grid_latitude']],),
                        }

    def draw(self, draw_method, *args, **kwargs):
        draw_fn = getattr(self, draw_method)
        draw_fn(*args, **kwargs)
        self.check_graphic()

    def run_tests(self, cube, results):
        for draw_method, coords in results:
            draw_method(cube, coords=coords)
            try:
                self.check_graphic()
            except AssertionError as err:
                self.fail('Draw method %r failed with coords: %r. '
                          'Assertion message: %s' % (draw_method, coords, err))

    def run_tests_1d(self, cube, results):
        # there is a different calling convention for 1d plots
        for draw_method, coords in results:
            draw_method(cube.coord(coords[0]), cube)
            try:
                self.check_graphic()
            except AssertionError as err:
                msg = 'Draw method {!r} failed with coords: {!r}. ' \
                      'Assertion message: {!s}'
                self.fail(msg.format(draw_method, coords, err))

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
        self.run_tests_1d(test_cube, self.results['x'])

    def test_y(self):
        test_cube = self.cube[0, 0, :, 0]
        self.run_tests_1d(test_cube, self.results['y'])

    def test_bad__duplicate_coord(self):
        cube = self.cube[0, 0, :, :]
        with self.assertRaises(ValueError):
            self.draw('contourf', cube, coords=['grid_longitude',
                                                'grid_longitude'])

    def test_bad__too_many_coords(self):
        cube = self.cube[0, 0, :, :]
        with self.assertRaises(ValueError):
            self.draw('contourf', cube, coords=['grid_longitude',
                                                'grid_longitude',
                                                'grid_latitude'])

    def test_bad__coord_name(self):
        cube = self.cube[0, 0, :, :]
        with self.assertRaises(CoordinateNotFoundError):
            self.draw('contourf', cube, coords=['grid_longitude', 'wibble'])

    def test_bad__no_coords_given(self):
        cube = self.cube[0, 0, :, :]
        with self.assertRaises(ValueError):
            self.draw('contourf', cube, coords=[])

    def test_bad__duplicate_coord_ref(self):
        cube = self.cube[0, 0, :, :]
        with self.assertRaises(ValueError):
            self.draw('contourf', cube, coords=[cube.coord('grid_longitude'),
                                                cube.coord('grid_longitude')])

    def test_bad__too_many_coord_refs(self):
        cube = self.cube[0, 0, :, :]
        with self.assertRaises(ValueError):
            self.draw('contourf', cube, coords=[cube.coord('grid_longitude'),
                                                cube.coord('grid_longitude'),
                                                cube.coord('grid_longitude')])

    def test_non_cube_coordinate(self):
        cube = self.cube[0, :, :, 0]
        pts = -100 + np.arange(cube.shape[1]) * 13
        x = coords.DimCoord(pts, standard_name='model_level_number',
                            attributes={'positive': 'up'})
        self.draw('contourf', cube, coords=['grid_latitude', x])
