# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.mask_cube"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris.tests.stock import (
    make_bounds_discontiguous_at_point,
    sample_2d_latlons,
)
from iris.util import mask_cube


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


if __name__ == "__main__":
    tests.main()
