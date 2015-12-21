# (C) British Crown Copyright 2013 - 2015, Met Office
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
Test function :func:`iris.util.cubelike_array_as_cube`.

FOR NOW: not proper tests, just exercising + not getting errors.
ALSO for now we are cheating and *also* testing cube.coord_as_cube here.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
import iris.tests
import iris.tests.stock as istk

from iris.util import cubelike_array_as_cube


class Test_cubelike_array_as_cube(iris.tests.IrisTest):
    pass

class Test_coord_as_cube(iris.tests.IrisTest):
    def setUp(self):
        self.cube_multidim = istk.simple_2d_w_multidim_coords()
        self.cube_3d = istk.realistic_3d()
        self.cubes = [self.cube_3d, self.cube_multidim]

    def test_allcoords(self):
        for i_test, cube in enumerate(self.cubes):
            print()
            print('Test #{}.  cube =...'.format(i_test))
            print(cube)
            for coord in cube.coords():
                print
                if cube.coord_dims(coord):
                    # Extract and print non-scalar coords.
                    coord_name = coord.name()
                    print('Extract {}:'.format(coord_name))
                    coord_cube = cube.coord_as_cube(coord_name)
                    print(coord_cube)

    def test1(self):
        print()
        print('New data over 1d, dim-0:')
        print(cubelike_array_as_cube(np.arange(3), self.cube_multidim, (0,),
                                     name='odd_data_dim0', units='m s-1'))
        print()
        print('New data over 1d, dim-1:')
        print(cubelike_array_as_cube(np.arange(4), self.cube_multidim, (1,),
                                     name='odd_data_dim1', units='K'))

        print()
        print('New data over 2d:')
        print(cubelike_array_as_cube(np.zeros((3, 4)),
                                     self.cube_multidim, (0, 1,),
                                     long_name='longname_2d', units='rad s-1'))

        print()
        print('Transposed new data over 2d:')
        print(cubelike_array_as_cube(
            np.zeros((4, 3)), self.cube_multidim, (1, 0,),
            standard_name='model_level_number', units='1'))

        print()
        print('Data over longitude+time:')
        print(cubelike_array_as_cube(np.zeros((11, 7)), self.cube_3d, (2, 0),
                                     name='twod_lon_time', units='m'))

        print()
        print('Data over time+longitude, specified by name:')
        print(cubelike_array_as_cube(np.zeros((7, 11)),
                                     self.cube_3d, ('time', 'grid_longitude'),
                                     name='twod_time_lons', units='m'))


if __name__ == '__main__':
    tests.main()
