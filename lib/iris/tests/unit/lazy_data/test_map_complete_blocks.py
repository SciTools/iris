# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris._lazy data.map_complete_blocks`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import unittest

import dask.array as da
import numpy as np

from iris._lazy_data import is_lazy_data, map_complete_blocks


def create_mock_cube(array):
    cube = unittest.mock.Mock()
    cube_data = unittest.mock.PropertyMock(return_value=array)
    type(cube).data = cube_data
    cube.dtype = array.dtype
    cube.has_lazy_data = unittest.mock.Mock(return_value=is_lazy_data(array))
    cube.lazy_data = unittest.mock.Mock(return_value=array)
    cube.shape = array.shape
    return cube, cube_data


class Test_map_complete_blocks(tests.IrisTest):
    def setUp(self):
        self.array = np.arange(8).reshape(2, 4)
        self.func = lambda chunk: chunk + 1
        self.func_result = self.array + 1

    def test_non_lazy_input(self):
        # Check that a non-lazy input doesn't trip up the functionality.
        cube, cube_data = create_mock_cube(self.array)
        result = map_complete_blocks(
            cube, self.func, dims=(1,), out_sizes=(4,)
        )
        self.assertFalse(is_lazy_data(result))
        self.assertArrayEqual(result, self.func_result)
        # check correct data was accessed
        cube.lazy_data.assert_not_called()
        cube_data.assert_called_once()

    def test_lazy_input(self):
        lazy_array = da.asarray(self.array, chunks=((1, 1), (4,)))
        cube, cube_data = create_mock_cube(lazy_array)
        result = map_complete_blocks(
            cube, self.func, dims=(1,), out_sizes=(4,)
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), self.func_result)
        # check correct data was accessed
        cube.lazy_data.assert_called_once()
        cube_data.assert_not_called()

    def test_rechunk(self):
        lazy_array = da.asarray(self.array, chunks=((1, 1), (2, 2)))
        cube, _ = create_mock_cube(lazy_array)
        result = map_complete_blocks(
            cube, self.func, dims=(1,), out_sizes=(4,)
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), self.func_result)

    def test_different_out_shape(self):
        lazy_array = da.asarray(self.array, chunks=((1, 1), (4,)))
        cube, _ = create_mock_cube(lazy_array)

        def func(_):
            return np.arange(2).reshape(1, 2)

        func_result = [[0, 1], [0, 1]]
        result = map_complete_blocks(cube, func, dims=(1,), out_sizes=(2,))
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), func_result)

    def test_multidimensional_input(self):
        array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        lazy_array = da.asarray(array, chunks=((1, 1), (1, 2), (4,)))
        cube, _ = create_mock_cube(lazy_array)
        result = map_complete_blocks(
            cube, self.func, dims=(1, 2), out_sizes=(3, 4)
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), array + 1)


if __name__ == "__main__":
    tests.main()
