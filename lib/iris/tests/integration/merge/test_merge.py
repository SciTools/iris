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

import numpy as np

from iris.coords import AuxCoord, DimCoord
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


class TestNaNs(tests.IrisTest):
    def test_merge_nan_coords(self):
        from sys import version_info

        from pkg_resources import parse_version

        # Test that nan valued coordinates merge together.
        cube1 = Cube(np.ones([3, 4]), "air_temperature", units="K")
        coord1 = DimCoord([1, 2, 3], long_name="x")
        coord2 = DimCoord([0, 1, 2, 3], long_name="y")
        nan_coord1 = AuxCoord(np.nan, long_name="nan1")
        nan_coord2 = AuxCoord([np.nan] * 4, long_name="nan2")
        cube1.add_dim_coord(coord1, 0)
        cube1.add_dim_coord(coord2, 1)
        cube1.add_aux_coord(nan_coord1)
        cube1.add_aux_coord(nan_coord2, 1)
        cubes = CubeList(cube1.slices_over("x"))
        cube2 = cubes.merge_cube()

        # Account for change in behaviour for py310+ when hashing a NaN
        # Reference https://github.com/SciTools/iris/pull/4874
        version = (
            f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        )
        if parse_version(version) >= parse_version("3.10.0"):
            # vector coordinate
            expected = np.array([np.nan] * cube1.shape[0])
        else:
            # scalar coordinate
            expected = nan_coord1.points

        self.assertArrayEqual(
            np.isnan(expected), np.isnan(cube2.coord("nan1").points)
        )
        self.assertArrayEqual(
            np.isnan(nan_coord2.points),
            np.isnan(cube2.coord("nan2").points),
        )


if __name__ == "__main__":
    tests.main()
