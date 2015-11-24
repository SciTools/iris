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
"""Integration tests for subset."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.coords import DimCoord
from iris.cube import Cube


def _make_test_cube():
    data = np.zeros((4, 4, 1))
    lats, longs = [0, 10, 20, 30], [5, 15, 25, 35]
    lat_coord = DimCoord(lats, standard_name='latitude', units='degrees')
    lon_coord = DimCoord(longs, standard_name='longitude', units='degrees')
    vrt_coord = DimCoord([850], long_name='pressure', units='hPa')
    return Cube(data,
                long_name='test_cube', units='1', attributes=None,
                dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)],
                aux_coords_and_dims=[(vrt_coord, None)])


class TestSubset(tests.IrisTest):
    def setUp(self):
        self.cube = _make_test_cube()

    def test_coordinate_subset(self):
        coord = self.cube.coord('pressure')
        subsetted = self.cube.subset(coord)
        self.assertEqual(cube, subsetted)


if __name__ == "__main__":
    tests.main()
