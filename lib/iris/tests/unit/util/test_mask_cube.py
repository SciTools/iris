# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.mask_cube."""

import pathlib

import dask.array as da
import numpy.ma as ma
import pytest

from iris.tests import _shared_utils
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
    @pytest.fixture(autouse=True)
    def _get_request(self, request):
        self.request = request

    def assert_original_metadata(self, cube, func):
        """Check metadata matches that of input cube.  func is a string indicating
        which function created the original cube.

        """
        reference_dir = pathlib.Path("unit/util/mask_cube")
        reference_fname = reference_dir / f"original_cube_{func}.cml"
        _shared_utils.assert_CML(
            self.request,
            cube,
            reference_filename=str(reference_fname),
            checksum=False,
        )


class TestArrayMask(MaskCubeMixin):
    """Tests with mask specified as numpy array."""

    @pytest.fixture(autouse=True)
    def _setup(self):
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
        _shared_utils.assert_array_equal(expected.data.mask, cube.data.mask)
        self.assert_original_metadata(cube, "full2d_global")
        assert returned is None

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
        _shared_utils.assert_array_equal(expected.data.mask, returned.data.mask)
        self.assert_original_metadata(returned, "full2d_global")
        assert not ma.is_masked(cube.data)

    def test_mask_cube_lazy_in_place_broadcast(self):
        cube = simple_2d()
        cube.data = cube.lazy_data()
        mask = [0, 1, 1, 0]
        returned = mask_cube(cube, mask, in_place=True)
        assert cube.has_lazy_data()
        # Touch the data so lazyness status doesn't affect CML check.
        cube.data
        self.assert_original_metadata(cube, "simple_2d")
        for subcube in cube.slices("foo"):
            # Mask should have been broadcast across "bar" dimension.
            _shared_utils.assert_array_equal(subcube.data.mask, mask)
        assert returned is None


class TestCoordMask(MaskCubeMixin):
    """Tests with mask specified as a Coord."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = simple_2d()

    def test_mask_cube_2d_first_dim(self):
        mask_coord = iris.coords.AuxCoord([0, 1, 0], long_name="mask", units=1)
        self.cube.add_aux_coord(mask_coord, 0)

        returned = mask_cube(self.cube, mask_coord, in_place=False)
        # Remove extra coord so we can check against original metadata.
        returned.remove_coord(mask_coord)
        self.assert_original_metadata(returned, "simple_2d")
        for subcube in returned.slices("bar"):
            # Mask should have been broadcast across "foo" dimension.
            _shared_utils.assert_array_equal(subcube.data.mask, mask_coord.points)

    def test_mask_cube_2d_second_dim(self):
        mask_coord = iris.coords.AuxCoord([0, 0, 1, 1], long_name="mask", units=1)
        returned = mask_cube(self.cube, mask_coord, in_place=False, dim=1)
        self.assert_original_metadata(returned, "simple_2d")
        for subcube in returned.slices("foo"):
            # Mask should have been broadcast across "bar" dimension.
            _shared_utils.assert_array_equal(subcube.data.mask, mask_coord.points)


class TestCubeMask(MaskCubeMixin):
    """Tests with mask specified as a Cube."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = simple_2d()

    def test_mask_cube_2d_first_dim_not_in_place(self):
        mask = iris.cube.Cube([0, 1, 0], long_name="mask", units=1)
        mask.add_dim_coord(self.cube.coord("bar"), 0)

        returned = mask_cube(self.cube, mask, in_place=False)
        self.assert_original_metadata(returned, "simple_2d")
        for subcube in returned.slices("bar"):
            # Mask should have been broadcast across 'foo' dimension.
            _shared_utils.assert_array_equal(subcube.data.mask, mask.data)

    def test_mask_cube_2d_first_dim_in_place(self):
        mask = iris.cube.Cube([0, 1, 0], long_name="mask", units=1)
        mask.add_dim_coord(self.cube.coord("bar"), 0)

        returned = mask_cube(self.cube, mask, in_place=True)
        self.assert_original_metadata(self.cube, "simple_2d")
        for subcube in self.cube.slices("bar"):
            # Mask should have been broadcast across 'foo' dimension.
            _shared_utils.assert_array_equal(subcube.data.mask, mask.data)
        assert returned is None

    def test_mask_cube_2d_create_new_dim(self):
        mask = iris.cube.Cube([[0, 1, 0], [0, 0, 1]], long_name="mask", units=1)

        broadcast_coord = iris.coords.DimCoord([1, 2], long_name="baz")
        mask.add_dim_coord(broadcast_coord, 0)
        mask.add_dim_coord(self.cube.coord("bar"), 1)

        # Create length-1 dimension to enable broadcasting.
        self.cube.add_aux_coord(broadcast_coord[0])
        cube = iris.util.new_axis(self.cube, "baz")

        returned = mask_cube(cube, mask, in_place=False)
        _shared_utils.assert_CML(self.request, cube, checksum=False)

        for subcube in returned.slices_over("baz"):
            # Underlying data should have been broadcast across 'baz' dimension.
            _shared_utils.assert_array_equal(subcube.data, self.cube.data)

        for subcube in returned.slices_over("foo"):
            # Mask should have been broadcast across 'foo' dimension.
            _shared_utils.assert_array_equal(subcube.data.mask, mask.data)

    def test_mask_cube_1d_lazy_mask_in_place(self):
        cube = simple_1d()
        mask = cube.copy(da.from_array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]))
        returned = mask_cube(cube, mask, in_place=True)
        assert returned is None
        assert cube.has_lazy_data()
        # Touch the data so lazyness status doesn't interfere with CML check.
        cube.data
        self.assert_original_metadata(cube, "simple_1d")
        _shared_utils.assert_array_equal(cube.data.mask, mask.data)
