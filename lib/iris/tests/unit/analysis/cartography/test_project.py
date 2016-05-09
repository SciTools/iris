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
"""Unit tests for :func:`iris.analysis.cartography.project`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import cartopy.crs as ccrs
import numpy as np

import iris.coord_systems
import iris.coords
import iris.cube
import iris.tests
import iris.tests.stock

from iris.analysis.cartography import project


ROBINSON = ccrs.Robinson()


def low_res_4d():
    cube = iris.tests.stock.realistic_4d_no_derived()
    cube = cube[0:2, 0:3, ::10, ::10]
    cube.remove_coord('surface_altitude')
    return cube


class TestAll(tests.IrisTest):
    def setUp(self):
        cs = iris.coord_systems.GeogCS(654321)
        self.cube = iris.cube.Cube(np.zeros(25).reshape(5, 5))
        self.cube.add_dim_coord(
            iris.coords.DimCoord(np.arange(5), standard_name="latitude",
                                 units='degrees', coord_system=cs), 0)
        self.cube.add_dim_coord(
            iris.coords.DimCoord(np.arange(5), standard_name="longitude",
                                 units='degrees', coord_system=cs), 1)

        self.tcs = iris.coord_systems.GeogCS(600000)

    def test_is_iris_coord_system(self):
        res, _ = project(self.cube, self.tcs)
        self.assertEqual(res.coord('projection_y_coordinate').coord_system,
                         self.tcs)
        self.assertEqual(res.coord('projection_x_coordinate').coord_system,
                         self.tcs)

        self.assertIsNot(res.coord('projection_y_coordinate').coord_system,
                         self.tcs)
        self.assertIsNot(res.coord('projection_x_coordinate').coord_system,
                         self.tcs)

    def test_bad_resolution_negative(self):
        cube = low_res_4d()
        with self.assertRaises(ValueError):
            project(cube, ROBINSON, nx=-200, ny=200)

    def test_bad_resolution_non_numeric(self):
        cube = low_res_4d()
        with self.assertRaises(ValueError):
            project(cube, ROBINSON, nx=200, ny='abc')

    def test_missing_lat(self):
        cube = low_res_4d()
        cube.remove_coord('grid_latitude')
        with self.assertRaises(ValueError):
            project(cube, ROBINSON)

    def test_missing_lon(self):
        cube = low_res_4d()
        cube.remove_coord('grid_longitude')
        with self.assertRaises(ValueError):
            project(cube, ROBINSON)

    def test_missing_latlon(self):
        cube = low_res_4d()
        cube.remove_coord('grid_longitude')
        cube.remove_coord('grid_latitude')
        with self.assertRaises(ValueError):
            project(cube, ROBINSON)

    def test_default_resolution(self):
        cube = low_res_4d()
        new_cube, extent = project(cube, ROBINSON)
        self.assertEqual(new_cube.shape, cube.shape)

    def test_mismatched_coord_systems(self):
        cube = low_res_4d()
        cube.coord('grid_longitude').coord_system = None
        with self.assertRaises(ValueError):
            project(cube, ROBINSON)

    @tests.skip_data
    def test_no_coord_system(self):
        cube = iris.load_cube(tests.get_data_path(('PP', 'aPPglob1',
                                                   'global.pp')))
        cube.coord('longitude').coord_system = None
        cube.coord('latitude').coord_system = None
        with iris.tests.mock.patch('warnings.warn') as warn:
            new_cube, extent = project(cube, ROBINSON)
        warn.assert_called_once_with('Coordinate system of latitude and '
                                     'longitude coordinates is not specified. '
                                     'Assuming WGS84 Geodetic.')
        self.assertCML(new_cube,
                       ('analysis', 'project', 'default_source_cs.cml'))


if __name__ == '__main__':
    tests.main()
