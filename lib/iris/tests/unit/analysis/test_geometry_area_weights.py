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
"""
Unit tests for the :func:`iris.analysis.geometry.geometry_area_weights`
function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from mock import Mock
import numpy as np
from shapely.geometry import Polygon

from iris.analysis.geometry import geometry_area_weights
from iris.coords import DimCoord
from iris.cube import Cube


class Test(tests.IrisTest):
    def setUp(self):
        x_coord = DimCoord([0.5, 1.5], 'longitude', bounds=[[0, 2], [2, 4]])
        y_coord = DimCoord([0.5, 1.5], 'latitude', bounds=[[0, 2], [2, 4]])
        self.data = np.empty((4, 2, 2))
        dim_coords_and_dims = [(y_coord, (1,)), (x_coord, (2,))]
        self.cube = Cube(self.data, dim_coords_and_dims=dim_coords_and_dims)
        self.geometry = Polygon([(3, 3), (3, 50), (50, 50), (50, 3)])

    def test_no_overlap(self):
        self.geometry = Polygon([(4, 4), (4, 6), (6, 6), (6, 4)])
        weights = geometry_area_weights(self.cube, self.geometry)
        self.assertEqual(np.sum(weights), 0)

    def test_overlap(self):
        weights = geometry_area_weights(self.cube, self.geometry)
        expected = np.repeat([[[0., 0.], [0., 1.]]], self.data.shape[0],
                             axis=0)
        self.assertArrayEqual(weights, expected)

    def test_overlap_normalize(self):
        weights = geometry_area_weights(self.cube, self.geometry,
                                        normalize=True)
        expected = np.repeat([[[0., 0.], [0., 0.25]]], self.data.shape[0],
                             axis=0)
        self.assertArrayEqual(weights, expected)


if __name__ == '__main__':
    tests.main()
