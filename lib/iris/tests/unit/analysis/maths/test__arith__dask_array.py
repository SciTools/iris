# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for cube arithmetic with dask arrays."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.

import dask.array as da

import iris.cube
from iris.tests.unit.analysis.maths import MathsAddOperationMixin


class TestArithDask(MathsAddOperationMixin):
    def test_compute_not_called(self, mocked_compute):
        # No data should be realised when adding a cube and a dask array.
        cube = iris.cube.Cube(da.arange(4))
        array = da.ones(4)

        self.data_op(cube, array)
        mocked_compute.assert_not_called()
