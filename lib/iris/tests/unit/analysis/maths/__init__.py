# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris.analysis.maths` module."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import ma

from iris.analysis import MEAN
from iris.coords import DimCoord
from iris.cube import Cube
import iris.tests.stock as stock


class CubeArithmeticBroadcastingTestMixin(metaclass=ABCMeta):
    # A framework for testing the broadcasting behaviour of the various cube
    # arithmetic operations.  (A test for each operation inherits this).
    @property
    @abstractmethod
    def data_op(self):
        # Define an operator to be called, I.E. 'operator.xx'.
        pass

    @property
    @abstractmethod
    def cube_func(self):
        # Define an iris arithmetic function to be called
        # I.E. 'iris.analysis.maths.xx'.
        pass

    def test_transposed(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.copy()
        other.transpose()
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        expected_data = self.data_op(cube.data, other.data.T)
        self.assertArrayEqual(res.data, expected_data)

    def test_collapse_zeroth_dim(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed("time", MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # No modification to other.data is needed as numpy broadcasting
        # should be sufficient.
        expected_data = self.data_op(cube.data, other.data)
        # Use assertMaskedArrayEqual as collapsing with MEAN results
        # in a cube with a masked data array.
        self.assertMaskedArrayEqual(res.data, expected_data)

    def test_collapse_all_dims(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed(cube.coords(dim_coords=True), MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # No modification to other.data is needed as numpy broadcasting
        # should be sufficient.
        expected_data = self.data_op(cube.data, other.data)
        # Use assertArrayEqual rather than assertMaskedArrayEqual as
        # collapsing all dims does not result in a masked array.
        self.assertArrayEqual(res.data, expected_data)

    def test_collapse_last_dims(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed(["grid_latitude", "grid_longitude"], MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # Transpose the dimensions in self.cube that have been collapsed in
        # other to lie at the front, thereby enabling numpy broadcasting to
        # function when applying data operator. Finish by transposing back
        # again to restore order.
        expected_data = self.data_op(
            cube.data.transpose((2, 3, 0, 1)), other.data
        ).transpose(2, 3, 0, 1)
        self.assertMaskedArrayEqual(res.data, expected_data)

    def test_collapse_middle_dim(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed(["model_level_number"], MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # Add the collapsed dimension back in via np.newaxis to enable
        # numpy broadcasting to function.
        expected_data = self.data_op(cube.data, other.data[:, np.newaxis, ...])
        self.assertMaskedArrayEqual(res.data, expected_data)

    def test_slice(self):
        cube = stock.realistic_4d_no_derived()
        for dim in range(cube.ndim):
            keys = [slice(None)] * cube.ndim
            keys[dim] = 3
            other = cube[tuple(keys)]
            res = self.cube_func(cube, other)
            self.assertCML(res, checksum=False)
            # Add the collapsed dimension back in via np.newaxis to enable
            # numpy broadcasting to function.
            keys[dim] = np.newaxis
            expected_data = self.data_op(cube.data, other.data[tuple(keys)])
            msg = "Problem broadcasting cubes when sliced on dimension {}."
            self.assertArrayEqual(
                res.data, expected_data, err_msg=msg.format(dim)
            )


class CubeArithmeticMaskingTestMixin(metaclass=ABCMeta):
    # A framework for testing the mask handling behaviour of the various cube
    # arithmetic operations.  (A test for each operation inherits this).
    @property
    @abstractmethod
    def data_op(self):
        # Define an operator to be called, I.E. 'operator.xx'.
        pass

    @property
    @abstractmethod
    def cube_func(self):
        # Define an iris arithmetic function to be called
        # I.E. 'iris.analysis.maths.xx'.
        pass

    def _test_partial_mask(self, in_place):
        # Helper method for masked data tests.
        dat_a = ma.array([2.0, 2.0, 2.0, 2.0], mask=[1, 0, 1, 0])
        dat_b = ma.array([2.0, 2.0, 2.0, 2.0], mask=[1, 1, 0, 0])

        cube_a = Cube(dat_a)
        cube_b = Cube(dat_b)

        com = self.data_op(dat_b, dat_a)
        res = self.cube_func(cube_b, cube_a, in_place=in_place)

        return com, res, cube_b

    def test_partial_mask_in_place(self):
        # Cube in_place arithmetic operation.
        com, res, orig_cube = self._test_partial_mask(True)

        self.assertMaskedArrayEqual(com, res.data, strict=True)
        self.assertIs(res, orig_cube)

    def test_partial_mask_not_in_place(self):
        # Cube arithmetic not an in_place operation.
        com, res, orig_cube = self._test_partial_mask(False)

        self.assertMaskedArrayEqual(com, res.data)
        self.assertIsNot(res, orig_cube)


class CubeArithmeticCoordsTest(tests.IrisTest):
    # This class sets up pairs of cubes to test iris' ability to reject
    # arithmetic operations on coordinates which do not match.
    def SetUpNonMatching(self):
        # On this cube pair, the coordinates to perform operations on do not
        # match in either points array or name.
        data = np.zeros((3, 4))
        a = DimCoord([1, 2, 3], long_name="a")
        b = DimCoord([1, 2, 3, 4], long_name="b")
        x = DimCoord([4, 5, 6], long_name="x")
        y = DimCoord([5, 6, 7, 8], long_name="y")

        nomatch1 = Cube(data, dim_coords_and_dims=[(a, 0), (b, 1)])
        nomatch2 = Cube(data, dim_coords_and_dims=[(x, 0), (y, 1)])

        return nomatch1, nomatch2

    def SetUpReversed(self):
        # On this cube pair, the coordinates to perform operations on have
        # matching long names but the points array on one cube is reversed
        # with respect to that on the other.
        data = np.zeros((3, 4))
        a1 = DimCoord([1, 2, 3], long_name="a")
        b1 = DimCoord([1, 2, 3, 4], long_name="b")
        a2 = DimCoord([3, 2, 1], long_name="a")
        b2 = DimCoord([1, 2, 3, 4], long_name="b")

        reversed1 = Cube(data, dim_coords_and_dims=[(a1, 0), (b1, 1)])
        reversed2 = Cube(data, dim_coords_and_dims=[(a2, 0), (b2, 1)])

        return reversed1, reversed2


class CubeArithmeticMaskedConstantTestMixin(metaclass=ABCMeta):
    @property
    @abstractmethod
    def cube_func(self):
        # Define an iris arithmetic function to be called
        # I.E. 'iris.analysis.maths.xx'.
        pass

    def test_masked_constant_in_place(self):
        # Cube in_place arithmetic operation.
        dtype = np.int64
        dat = ma.masked_array(0, 1, dtype)
        cube = Cube(dat)
        res = self.cube_func(cube, 5, in_place=True)
        self.assertMaskedArrayEqual(ma.masked_array(0, 1), res.data)
        self.assertEqual(dtype, res.dtype)
        self.assertIs(res, cube)

    def test_masked_constant_not_in_place(self):
        # Cube in_place arithmetic operation.
        dtype = np.int64
        dat = ma.masked_array(0, 1, dtype)
        cube = Cube(dat)
        res = self.cube_func(cube, 5, in_place=False)
        self.assertMaskedArrayEqual(ma.masked_array(0, 1), res.data)
        self.assertEqual(dtype, res.dtype)
        self.assertIsNot(res, cube)
