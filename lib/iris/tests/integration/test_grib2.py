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
"""Integration tests for loading and saving GRIB2 files."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy.ma as ma

from iris import FUTURE, load_cube


@tests.skip_data
class TestImport(tests.IrisTest):
    def test_gdt1(self):
        with FUTURE.context(strict_grib_load=True):
            path = tests.get_data_path(('GRIB', 'rotated_nae_t',
                                        'sensible_pole.grib2'))
            cube = load_cube(path)
            self.assertCMLApproxData(cube)

    def test_gdt90_with_bitmap(self):
        with FUTURE.context(strict_grib_load=True):
            path = tests.get_data_path(('GRIB', 'umukv', 'ukv_chan9.grib2'))
            cube = load_cube(path)
            # Pay particular attention to the orientation.
            self.assertIsNot(cube.data[0, 0], ma.masked)
            self.assertIs(cube.data[-1, 0], ma.masked)
            self.assertIs(cube.data[0, -1], ma.masked)
            self.assertIs(cube.data[-1, -1], ma.masked)
            x = cube.coord('projection_x_coordinate').points
            y = cube.coord('projection_y_coordinate').points
            self.assertGreater(x[0], x[-1])  # Decreasing X coordinate
            self.assertLess(y[0], y[-1])  # Increasing Y coordinate
            # Check everything else.
            self.assertCMLApproxData(cube)


@tests.skip_data
class TestPDT8(tests.IrisTest):
    def setUp(self):
        # Load from the test file.
        file_path = tests.get_data_path(('GRIB', 'time_processed',
                                         'time_bound.grib2'))
        with FUTURE.context(strict_grib_load=True):
            self.cube = load_cube(file_path)

    def test_coords(self):
        # Check the result has main coordinates as expected.
        for name, shape, is_bounded in [
                ('forecast_reference_time', (1,), False),
                ('time', (1,), True),
                ('forecast_period', (1,), True),
                ('pressure', (1,), False),
                ('latitude', (73,), False),
                ('longitude', (96,), False)]:
            coords = self.cube.coords(name)
            self.assertEqual(len(coords), 1,
                             'expected one {!r} coord, found {}'.format(
                                 name, len(coords)))
            coord, = coords
            self.assertEqual(coord.shape, shape,
                             'coord {!r} shape is {} instead of {!r}.'.format(
                                 name, coord.shape, shape))
            self.assertEqual(coord.has_bounds(), is_bounded,
                             'coord {!r} has_bounds={}, expected {}.'.format(
                                 name, coord.has_bounds(), is_bounded))

    def test_cell_method(self):
        # Check the result has the expected cell method.
        cell_methods = self.cube.cell_methods
        self.assertEqual(len(cell_methods), 1,
                         'result has {} cell methods, expected one.'.format(
                             len(cell_methods)))
        cell_method, = cell_methods
        self.assertEqual(cell_method.coord_names, ('time',))


if __name__ == '__main__':
    tests.main()
