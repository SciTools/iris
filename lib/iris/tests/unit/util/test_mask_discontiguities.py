# (C) British Crown Copyright 2018, Met Office
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
"""Unit tests for handling and plotting of 2-dimensional coordinates"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.util import mask_data_at_discontiguities
from iris.tests.stock import simple_3d
from iris.tests.stock import sample_2d_latlons
from iris.tests.stock import make_bounds_discontiguous_at_point
from iris.tests.stock._stock_2d_latlons import grid_coords_2d_from_1d


def full2d_global():
    return sample_2d_latlons(transformed=True)


@tests.skip_data
class Test(tests.IrisTest):
    def setUp(self):
        # Set up a 2d cube with a masked discontiguity to test masking
        # of 2-dimensional cubes
        self.cube_2d = full2d_global()
        make_bounds_discontiguous_at_point(self.cube_2d, 3, 3)

        # Construct a 3d cube with a different masked discontiguity to test
        # masking of multidimensional cubes
        self.cube_3d = simple_3d()
        x_coord = self.cube_3d.coord('longitude')
        x_coord.guess_bounds()
        y_coord = self.cube_3d.coord('latitude')
        y_coord.guess_bounds()
        x_coord_2d, y_coord_2d = grid_coords_2d_from_1d(x_coord, y_coord)
        # Remove the old grid coords.
        for coord in (x_coord, y_coord):
            self.cube_3d.remove_coord(coord)
        # Add the new grid coords.
        for coord in (x_coord_2d, y_coord_2d):
            self.cube_3d.add_aux_coord(coord, (1, 2))
        # Add a discontiguity and mask the data point
        make_bounds_discontiguous_at_point(self.cube_3d, 0, 0)

    def test_mask_discontiguities_2d(self):
        # This tests the masking of a 2d data array
        cube = self.cube_2d
        coord = cube.coord('longitude')
        expected = cube.copy()

        # Remove mask so that we can pass an unmasked data set to
        # mask_discontiguities, and check that it masks the correct point by
        # comparing with masked data
        cube.data.mask = ma.nomask
        returned = mask_data_at_discontiguities(cube, coord)
        self.assertTrue(np.all(expected.data.mask == returned.data.mask))

    def test_mask_discontiguities_3d(self):
        # This tests the broadcasting of the mask array from 2d to 3d
        cube = self.cube_3d
        coord = cube.coord('longitude')
        expected = cube.copy()

        # Remove mask so that we can pass an unmasked data set to
        # mask_discontiguities, and check that it masks the correct point by
        # comparing with masked data
        cube.data.mask = ma.nomask
        returned = mask_data_at_discontiguities(cube, coord)
        self.assertTrue(np.all(expected.data.mask == returned.data.mask))


if __name__ == '__main__':
    tests.main()
