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
"""
Integration tests for concatenating cubes with differing time coord epochs
using :func:`iris.util.unify_time_units`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import iris.coords
from iris._concatenate import concatenate
import iris.cube
import iris.unit
from iris.util import unify_time_units


class Test_concatenate__epoch(tests.IrisTest):
    def simple_1d_time_cubes(self, reftimes, coords_points):
        cubes = []
        data_points = [273, 275, 278, 277, 274]
        for reftime, coord_points in zip(reftimes, coords_points):
            cube = iris.cube.Cube(np.array(data_points, dtype=np.float32),
                                  standard_name='air_temperature',
                                  units='K')
            unit = iris.unit.Unit(reftime, calendar='gregorian')
            coord = iris.coords.DimCoord(points=np.array(coord_points,
                                                         dtype=np.float32),
                                         standard_name='time',
                                         units=unit)
            cube.add_dim_coord(coord, 0)
            cubes.append(cube)
        return cubes

    def test_concat_1d_with_differing_time_units(self):
        reftimes = ['hours since 1970-01-01 00:00:00',
                    'hours since 1970-01-02 00:00:00']
        coords_points = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        cubes = self.simple_1d_time_cubes(reftimes, coords_points)
        unify_time_units(cubes)
        result = concatenate(cubes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10,))


if __name__ == '__main__':
    tests.main()
