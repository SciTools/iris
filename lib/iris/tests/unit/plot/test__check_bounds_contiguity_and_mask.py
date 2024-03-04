# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot._check_bounds_contiguity_and_mask`
function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np
import numpy.ma as ma

from iris.coords import DimCoord
from iris.plot import _check_bounds_contiguity_and_mask
from iris.tests.stock import (
    make_bounds_discontiguous_at_point,
    sample_2d_latlons,
)


@tests.skip_plot
class Test_check_bounds_contiguity_and_mask(tests.IrisTest):
    def test_1d_not_checked(self):
        # Test a 1D coordinate, which is not checked as atol is not set.
        coord = DimCoord([1, 3, 5], bounds=[[0, 2], [2, 4], [5, 6]])
        data = np.array([278, 300, 282])
        # Make sure contiguity checking doesn't throw an error
        _check_bounds_contiguity_and_mask(coord, data)

    def test_1d_contiguous(self):
        # Test that a 1D coordinate which is contiguous does not fail.
        coord = DimCoord([1, 3, 5], bounds=[[0, 2], [2, 4], [4, 6]])
        data = np.array([278, 300, 282])
        _check_bounds_contiguity_and_mask(coord, data, atol=1e-3)

    def test_1d_discontigous_masked(self):
        # Test a 1D coordinate which is discontiguous but masked at
        # discontiguities.
        coord = DimCoord([1, 3, 5], bounds=[[0, 2], [2, 4], [5, 6]])
        data = ma.array(np.array([278, 300, 282]), mask=[0, 1, 0])
        _check_bounds_contiguity_and_mask(coord, data, atol=1e-3)

    def test_1d_discontigous_unmasked(self):
        # Test a 1D coordinate which is discontiguous and unmasked at
        # discontiguities.
        coord = DimCoord([1, 3, 5], bounds=[[0, 2], [2, 4], [5, 6]])
        data = ma.array(np.array([278, 300, 282]), mask=[1, 0, 0])
        msg = (
            "coordinate are not contiguous and data is not masked where "
            "the discontiguity occurs"
        )
        with self.assertRaisesRegex(ValueError, msg):
            _check_bounds_contiguity_and_mask(coord, data, atol=1e-3)

    def test_2d_contiguous(self):
        # Test that a 2D coordinate which is contiguous does not throw
        # an error.
        cube = sample_2d_latlons()
        _check_bounds_contiguity_and_mask(cube.coord("longitude"), cube.data)

    def test_2d_contiguous_atol(self):
        # Check the atol is passed correctly.
        cube = sample_2d_latlons()
        with mock.patch(
            "iris.coords.Coord._discontiguity_in_bounds"
        ) as discontiguity_check:
            # Discontiguity returns two objects that are unpacked in
            # `_check_bounds_contiguity_and_mask`.
            discontiguity_check.return_value = [True, None]
            _check_bounds_contiguity_and_mask(
                cube.coord("longitude"), cube.data, atol=1e-3
            )
        discontiguity_check.assert_called_with(atol=1e-3)

    def test_2d_discontigous_masked(self):
        # Test that a 2D coordinate which is discontiguous but masked at
        # discontiguities doesn't error.
        cube = sample_2d_latlons()
        make_bounds_discontiguous_at_point(cube, 3, 4)
        _check_bounds_contiguity_and_mask(cube.coord("longitude"), cube.data)

    def test_2d_discontigous_unmasked(self):
        # Test a 2D coordinate which is discontiguous and unmasked at
        # discontiguities.
        cube = sample_2d_latlons()
        make_bounds_discontiguous_at_point(cube, 3, 4)
        msg = "coordinate are not contiguous"
        cube.data[3, 4] = ma.nomask
        with self.assertRaisesRegex(ValueError, msg):
            _check_bounds_contiguity_and_mask(
                cube.coord("longitude"), cube.data
            )


if __name__ == "__main__":
    tests.main()
