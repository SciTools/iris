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
"""Test function :func:`iris.util.mask_cube"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.util import mask_cube
from iris.tests.stock import sample_2d_latlons
from iris.tests.stock import make_bounds_discontiguous_at_point


def full2d_global():
    return sample_2d_latlons(transformed=True)


@tests.skip_data
class Test(tests.IrisTest):
    def setUp(self):
        # Set up a 2d cube with a masked discontiguity to test masking
        # of 2-dimensional cubes
        self.cube_2d = full2d_global()
        make_bounds_discontiguous_at_point(self.cube_2d, 3, 3)

    def test_mask_cube_2d(self):
        # This tests the masking of a 2d data array
        cube = self.cube_2d
        discontiguity_array = ma.getmaskarray(cube.data).copy()
        expected = cube.copy()

        # Remove mask so that we can pass an unmasked data set to
        # mask_discontiguities, and check that it masks the correct point by
        # comparing with masked data
        cube.data.mask = ma.nomask
        returned = mask_cube(cube, discontiguity_array)
        self.assertTrue(np.all(expected.data.mask == returned.data.mask))


if __name__ == '__main__':
    tests.main()
