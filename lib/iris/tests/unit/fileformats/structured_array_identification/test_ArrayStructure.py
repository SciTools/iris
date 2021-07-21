# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the
:mod:`iris.fileformats._structured_array_identification.ArrayStructure` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.fileformats._structured_array_identification import (
    ArrayStructure,
    _UnstructuredArrayException,
)


def construct_nd(sub_array, sub_dim, shape):
    # Given a 1D array, a shape, and the axis/dimension that the 1D array
    # represents on the bigger array, construct a numpy array which is
    # filled appropriately.
    assert sub_array.ndim == 1
    sub_shape = [1 if dim != sub_dim else -1 for dim in range(len(shape))]
    return sub_array.reshape(sub_shape) * np.ones(shape)


class TestArrayStructure_from_array(tests.IrisTest):
    def struct_from_arr(self, nd_array):
        return ArrayStructure.from_array(nd_array.flatten())

    def test_1d_len_0(self):
        a = np.arange(0)
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, a))

    def test_1d_len_1(self):
        a = np.arange(1)
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, a))

    def test_1d(self):
        a = np.array([-1, 3, 1, 2])
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, a))

    def test_1d_ones(self):
        a = np.ones(10)
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, [1]))

    def test_1d_range(self):
        a = np.arange(6)
        self.assertEqual(
            self.struct_from_arr(a), ArrayStructure(1, list(range(6)))
        )

    def test_3d_ones(self):
        a = np.ones([10, 2, 1])
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, [1]))

    def test_1d_over_2d_first_dim_manual(self):
        sub = np.array([10, 10, 20, 20])
        self.assertEqual(
            self.struct_from_arr(sub), ArrayStructure(2, [10, 20])
        )

    def test_3d_first_dimension(self):
        flattened = np.array([1, 1, 1, 2, 2, 2])
        self.assertEqual(
            ArrayStructure.from_array(flattened), ArrayStructure(3, [1, 2])
        )

    def test_1d_over_2d_first_dim(self):
        sub = np.array([-1, 3, 1, 2])
        a = construct_nd(sub, 0, (4, 2))
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(2, sub))

    def test_1d_over_2d_second_dim(self):
        sub = np.array([-1, 3, 1, 2])
        a = construct_nd(sub, 1, (2, 4))
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, sub))

    def test_1d_over_3d_first_dim(self):
        sub = np.array([-1, 3, 1, 2])
        a = construct_nd(sub, 0, (4, 2, 3))
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(6, sub))

    def test_1d_over_3d_second_dim(self):
        sub = np.array([-1, 3, 1, 2])
        a = construct_nd(sub, 1, (2, 4, 3))
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(3, sub))

    def test_1d_over_3d_third_dim(self):
        sub = np.array([-1, 3, 1, 2])
        a = construct_nd(sub, 2, (3, 2, 4))
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, sub))

    def test_irregular_3d(self):
        sub = np.array([-1, 3, 1, 2])
        a = construct_nd(sub, 2, (3, 2, 4))
        a[0, 0, 0] = 5
        self.assertEqual(self.struct_from_arr(a), None)

    def test_repeated_3d(self):
        sub = np.array([-1, 3, 1, 2])
        a = construct_nd(sub, 2, (3, 2, 4))
        a[:, 0, 0] = 1
        self.assertEqual(self.struct_from_arr(a), None)

    def test_rolled_3d(self):
        # Shift the 3D array on by one, making the array 1d.
        sub = np.arange(4)
        a = construct_nd(sub, 0, (4, 2, 3))
        a = np.roll(a.flatten(), 1)
        self.assertEqual(self.struct_from_arr(a), None)

    def test_len_1_3d(self):
        # Setup a case which triggers an IndexError when identifying
        # the stride, but the result should still be correct.
        sub = np.arange(2)
        a = construct_nd(sub, 1, (1, 1, 1))
        self.assertEqual(self.struct_from_arr(a), ArrayStructure(1, sub))

    def test_not_an_array(self):
        # Support lists as an argument.
        self.assertEqual(
            ArrayStructure.from_array([1, 2, 3]), ArrayStructure(1, [1, 2, 3])
        )

    def test_multi_dim_array(self):
        with self.assertRaises(ValueError):
            ArrayStructure.from_array(np.arange(12).reshape(3, 4))


class nd_array_and_dims_cases:
    """
    Defines the test functionality for nd_array_and_dims. This class
    isn't actually the test case - see the C order and F order subclasses
    for those.

    """

    def test_scalar_len1_first_dim(self):
        struct = ArrayStructure(1, [1])
        orig = np.array([1, 1, 1])

        array, dims = struct.nd_array_and_dims(orig, (1, 3), order=self.order)
        self.assertArrayEqual(array, [1])
        self.assertEqual(dims, ())

    def test_scalar_non_len1_first_dim(self):
        struct = ArrayStructure(1, [1])
        orig = np.array([1, 1, 1])

        array, dims = struct.nd_array_and_dims(orig, (3, 1), order=self.order)
        self.assertArrayEqual(array, [1])
        self.assertEqual(dims, ())

    def test_single_vector(self):
        orig = construct_nd(np.array([1, 2]), 0, (2, 1, 3))
        flattened = orig.flatten(order=self.order)
        struct = ArrayStructure.from_array(flattened)
        array, dims = struct.nd_array_and_dims(
            flattened, (2, 1, 3), order=self.order
        )
        self.assertArrayEqual(array, [1, 2])
        self.assertEqual(dims, (0,))

    def test_single_vector_3rd_dim(self):
        orig = construct_nd(np.array([1, 2, 3]), 2, (4, 1, 3))
        flattened = orig.flatten(order=self.order)

        struct = ArrayStructure.from_array(flattened)
        array, dims = struct.nd_array_and_dims(
            flattened, (4, 1, 3), order=self.order
        )
        self.assertArrayEqual(array, [1, 2, 3])
        self.assertEqual(dims, (2,))

    def test_orig_array_and_target_shape_inconsistent(self):
        # An array structure which has a length which is a product
        # of potential dimensions should not result in an array
        struct = ArrayStructure(2, [1, 2, 3])
        orig = np.array([1, 1, 2, 2, 3, 3])

        msg = "Original array and target shape do not match up."
        with self.assertRaisesRegex(ValueError, msg):
            struct.nd_array_and_dims(orig, (2, 3, 2), order=self.order)

    def test_array_bigger_than_expected(self):
        # An array structure which has a length which is a product
        # of potential dimensions should not result in an array
        struct = ArrayStructure(2, [1, 2, 3, 4, 5, 6])
        orig = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

        with self.assertRaises(_UnstructuredArrayException):
            struct.nd_array_and_dims(orig, (2, 3, 2), order=self.order)

    def test_single_vector_extra_dimension(self):
        orig = construct_nd(np.array([1, 2]), 1, (3, 2))
        flattened = orig.flatten(order=self.order)

        struct = ArrayStructure.from_array(flattened)

        # Add another dimension on flattened, making it a (6, 2).
        input_array = np.vstack([flattened, flattened + 100]).T

        array, dims = struct.nd_array_and_dims(
            input_array, (3, 1, 2, 1), order=self.order
        )
        self.assertArrayEqual(array, [[1, 101], [2, 102]])
        self.assertEqual(dims, (2,))


class TestArrayStructure_nd_array_and_dims_f_order(
    tests.IrisTest, nd_array_and_dims_cases
):
    order = "f"


class TestArrayStructure_nd_array_and_dims_c_order(
    tests.IrisTest, nd_array_and_dims_cases
):
    order = "c"


if __name__ == "__main__":
    tests.main()
