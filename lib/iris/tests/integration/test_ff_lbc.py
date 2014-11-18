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
"""Integration tests for loading LBC fieldsfiles."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris import load as iris_load_cubes


class TestLBC(tests.IrisTest):
    def setUp(self):
        # Load multiple cubes from a test file.
        file_path = tests.get_data_path(('FF', 'lbc', 'small_lbc'))
        self.cubes = iris_load_cubes(file_path)

    def test_coords(self):
        # Check a few aspects of the loaded cubes.
        cubes = self.cubes
        self.assertEqual(len(cubes), 10)
        self.assertEqual(cubes[0].shape, (16, 16))
        self.assertEqual(cubes[1].shape, (2, 4, 16, 16))
        self.assertEqual(cubes[3].shape, (2, 5, 16, 16))

        # Check coordinates of one cube.
        cube = cubes[1]
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

        # Check a few data points as well.
        self.assertArrayAllClose(
            cube.data[:, ::2, 6, 13],
            np.array([[4.218922, 10.074577],
                      [4.626897, 6.520156]]),
            atol=1.0e-6)

        # Check centre 6x2 section (only) is masked.
        mask = np.zeros((2, 4, 16, 16), dtype=bool)
        mask[:, :, 7:9, 5:11] = True
        self.assertArrayEqual(cube.data.mask, mask)


if __name__ == '__main__':
    tests.main()
