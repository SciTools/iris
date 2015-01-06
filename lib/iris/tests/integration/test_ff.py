# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Integration tests for loading LBC fieldsfiles."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import shutil

import mock
import numpy as np

import iris
import iris.experimental.um as um


@tests.skip_data
class TestLBC(tests.IrisTest):
    def setUp(self):
        # Load multiple cubes from a test file.
        file_path = tests.get_data_path(('FF', 'lbc', 'small_lbc'))
        self.all_cubes = iris.load(file_path)
        # Select the second cube for detailed checks (the first is orography).
        self.test_cube = self.all_cubes[1]

    def test_various_cubes_shapes(self):
        # Check a few aspects of the loaded cubes.
        cubes = self.all_cubes
        self.assertEqual(len(cubes), 10)
        self.assertEqual(cubes[0].shape, (16, 16))
        self.assertEqual(cubes[1].shape, (2, 4, 16, 16))
        self.assertEqual(cubes[3].shape, (2, 5, 16, 16))

    def test_cube_coords(self):
        # Check coordinates of one cube.
        cube = self.test_cube
        self.assertEqual(len(cube.coords()), 8)
        for name, shape in [
                ('forecast_reference_time', (1,)),
                ('time', (2,)),
                ('forecast_period', (2,)),
                ('model_level_number', (4,)),
                ('level_height', (1,)),
                ('sigma', (1,)),
                ('grid_latitude', (16,)),
                ('grid_longitude', (16,))]:
            coords = cube.coords(name)
            self.assertEqual(len(coords), 1,
                             'expected one {!r} coord, found {}'.format(
                                 name, len(coords)))
            coord, = coords
            self.assertEqual(coord.shape, shape,
                             'coord {!r} shape is {} instead of {!r}.'.format(
                                 name, coord.shape, shape))

    def test_cube_data(self):
        # Check just a few points of the data.
        cube = self.test_cube
        self.assertArrayAllClose(
            cube.data[:, ::2, 6, 13],
            np.array([[4.218922, 10.074577],
                      [4.626897, 6.520156]]),
            atol=1.0e-6)

    def test_cube_mask(self):
        # Check the data mask : should be just the centre 6x2 section.
        cube = self.test_cube
        mask = np.zeros((2, 4, 16, 16), dtype=bool)
        mask[:, :, 7:9, 5:11] = True
        self.assertArrayEqual(cube.data.mask, mask)


class TestFFGrid(tests.IrisTest):
    @tests.skip_data
    def test_unhandled_grid_type(self):
        self.filename = tests.get_data_path(('FF', 'n48_multi_field'))
        with self.temp_filename() as temp_path:
            shutil.copyfile(self.filename, temp_path)
            ffv = um.FieldsFileVariant(temp_path,
                                       mode=um.FieldsFileVariant.UPDATE_MODE)
            ffv.fields[3].lbuser4 = 60
            ffv.close()
            with mock.patch('warnings.warn') as warn_fn:
                iris.load(temp_path)
            self.assertIn("Assuming the data is on a P grid.",
                          warn_fn.call_args[0][0])


if __name__ == '__main__':
    tests.main()
