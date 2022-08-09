# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.mask_cube"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import pathlib

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris.tests.stock import (
    make_bounds_discontiguous_at_point,
    sample_2d_latlons,
    simple_1d,
    simple_2d,
)
import iris.util
from iris.util import mask_cube


def full2d_global():
    return sample_2d_latlons(transformed=True)


class MaskCubeMixin:
    def assertOriginalMetadata(self, cube, func):
        """
        Check metadata matches that of input cube.  func is a string indicating
        which function created the original cube.

        """
        reference_dir = pathlib.Path("unit/util/mask_cube")
        reference_fname = reference_dir / f"original_cube_{func}.cml"
        self.assertCML(
            cube,
            reference_filename=str(reference_fname),
            checksum=False,
        )


class TestArrayMask(tests.IrisTest, MaskCubeMixin):
    """Tests with mask specified as numpy array."""

    def setUp(self):
        # Set up a 2d cube with a masked discontiguity to test masking
        # of 2-dimensional cubes
        self.cube_2d = full2d_global()
        make_bounds_discontiguous_at_point(self.cube_2d, 3, 3)

    def test_mask_cube_2d_in_place(self):
        # This tests the masking of a 2d data array
        cube = self.cube_2d
        discontiguity_array = ma.getmaskarray(cube.data).copy()
        expected = cube.copy()

        # Remove mask so that we can pass an unmasked data set to
        # mask_discontiguities, and check that it masks the correct point by
        # comparing with masked data
        cube.data = cube.data.data
        returned = mask_cube(cube, discontiguity_array, in_place=True)
        np.testing.assert_array_equal(expected.data.mask, cube.data.mask)
        self.assertOriginalMetadata(cube, "full2d_global")
        self.assertIs(returned, None)

    def test_mask_cube_2d_not_in_place(self):
        # This tests the masking of a 2d data array
        cube = self.cube_2d
        discontiguity_array = ma.getmaskarray(cube.data).copy()
        expected = cube.copy()

        # Remove mask so that we can pass an unmasked data set to
        # mask_discontiguities, and check that it masks the correct point by
        # comparing with masked data
        cube.data = cube.data.data
        returned = mask_cube(cube, discontiguity_array, in_place=False)
        np.testing.assert_array_equal(expected.data.mask, returned.data.mask)
        self.assertOriginalMetadata(returned, "full2d_global")
        self.assertFalse(ma.is_masked(cube.data))

    def test_mask_cube_lazy_in_place_broadcast(self):
        cube = simple_2d()
        cube.data = cube.lazy_data()
        mask = [0, 1, 1, 0]
        returned = mask_cube(cube, mask, in_place=True)
        self.assertTrue(cube.has_lazy_data())
        # Touch the data so lazyness status doesn't affect CML check.
        cube.data
        self.assertOriginalMetadata(cube, "simple_2d")
        for subcube in cube.slices("foo"):
            # Mask should have been broadcast across "bar" dimension.
            np.testing.assert_array_equal(subcube.data.mask, mask)
        self.assertIs(returned, None)


class TestCoordMask(tests.IrisTest, MaskCubeMixin):
    """Tests with mask specified as a Coord."""

    def setUp(self):
        self.cube = simple_2d()

    def test_mask_cube_2d_first_dim(self):
        mask_coord = iris.coords.AuxCoord([0, 1, 0], long_name="mask", units=1)
        self.cube.add_aux_coord(mask_coord, 0)

        returned = mask_cube(self.cube, mask_coord, in_place=False)
        # Remove extra coord so we can check against original metadata.
        returned.remove_coord(mask_coord)
        self.assertOriginalMetadata(returned, "simple_2d")
        for subcube in returned.slices("bar"):
            # Mask should have been broadcast across "foo" dimension.
            np.testing.assert_array_equal(subcube.data.mask, mask_coord.points)

    def test_mask_cube_2d_second_dim(self):
        mask_coord = iris.coords.AuxCoord(
            [0, 0, 1, 1], long_name="mask", units=1
        )
        returned = mask_cube(self.cube, mask_coord, in_place=False, dim=1)
        self.assertOriginalMetadata(returned, "simple_2d")
        for subcube in returned.slices("foo"):
            # Mask should have been broadcast across "bar" dimension.
            np.testing.assert_array_equal(subcube.data.mask, mask_coord.points)


class TestCubeMask(tests.IrisTest, MaskCubeMixin):
    """Tests with mask specified as a Cube."""

    def setUp(self):
        self.cube = simple_2d()

    def test_mask_cube_2d_first_dim_not_in_place(self):
        mask = iris.cube.Cube([0, 1, 0], long_name="mask", units=1)
        mask.add_dim_coord(self.cube.coord("bar"), 0)

        returned = mask_cube(self.cube, mask, in_place=False)
        self.assertOriginalMetadata(returned, "simple_2d")
        for subcube in returned.slices("bar"):
            # Mask should have been broadcast across 'foo' dimension.
            np.testing.assert_array_equal(subcube.data.mask, mask.data)

    def test_mask_cube_2d_first_dim_in_place(self):
        mask = iris.cube.Cube([0, 1, 0], long_name="mask", units=1)
        mask.add_dim_coord(self.cube.coord("bar"), 0)

        returned = mask_cube(self.cube, mask, in_place=True)
        self.assertOriginalMetadata(self.cube, "simple_2d")
        for subcube in self.cube.slices("bar"):
            # Mask should have been broadcast across 'foo' dimension.
            np.testing.assert_array_equal(subcube.data.mask, mask.data)
        self.assertIs(returned, None)

    def test_mask_cube_2d_create_new_dim(self):
        mask = iris.cube.Cube(
            [[0, 1, 0], [0, 0, 1]], long_name="mask", units=1
        )

        broadcast_coord = iris.coords.DimCoord([1, 2], long_name="baz")
        mask.add_dim_coord(broadcast_coord, 0)
        mask.add_dim_coord(self.cube.coord("bar"), 1)

        # Create length-1 dimension to enable broadcasting.
        self.cube.add_aux_coord(broadcast_coord[0])
        cube = iris.util.new_axis(self.cube, "baz")

        returned = mask_cube(cube, mask, in_place=False)
        self.assertCML(cube, checksum=False)

        for subcube in returned.slices_over("baz"):
            # Underlying data should have been broadcast across 'baz' dimension.
            np.testing.assert_array_equal(subcube.data, self.cube.data)

        for subcube in returned.slices_over("foo"):
            # Mask should have been broadcast across 'foo' dimension.
            np.testing.assert_array_equal(subcube.data.mask, mask.data)

    def test_mask_cube_1d_lazy_mask_in_place(self):
        cube = simple_1d()
        mask = cube.copy(da.from_array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]))
        returned = mask_cube(cube, mask, in_place=True)
        self.assertIs(returned, None)
        self.assertTrue(cube.has_lazy_data())
        # Touch the data so lazyness status doesn't interfere with CML check.
        cube.data
        self.assertOriginalMetadata(cube, "simple_1d")
        np.testing.assert_array_equal(cube.data.mask, mask.data)


if __name__ == "__main__":
    tests.main()
