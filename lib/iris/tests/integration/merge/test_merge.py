# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests for merging cubes.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

from iris.coords import DimCoord
from iris.cube import Cube, CubeList


class TestContiguous(tests.IrisTest):
    def test_form_contiguous_dimcoord(self):
        # Test that cube sliced up and remerged in the opposite order maintains
        # contiguity.
        cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        coord1 = DimCoord([3, 2, 1], long_name="spam")
        coord1.guess_bounds()
        cube1.add_dim_coord(coord1, 0)
        cubes = CubeList(cube1.slices_over("spam"))
        cube2 = cubes.merge_cube()
        coord2 = cube2.coord("spam")

        self.assertTrue(coord2.is_contiguous())
        self.assertArrayEqual(coord2.points, [1, 2, 3])
        self.assertArrayEqual(coord2.bounds, coord1.bounds[::-1, ::-1])


if __name__ == "__main__":
    tests.main()
