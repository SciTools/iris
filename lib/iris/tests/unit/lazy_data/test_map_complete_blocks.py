# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
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
    # Remove compute so cube is not interpreted as dask array.
    del cube.compute
    return cube, cube_data


class Test_map_complete_blocks(tests.IrisTest):
    def setUp(self):
        self.array = np.arange(8).reshape(2, 4)

        def func(chunk):
            """Use a function that cannot be 'sampled'.

            To make sure the call to map_blocks is correct for any function,
            we define this function that cannot be called with size 0 arrays
            to infer the output meta.
            """
            if chunk.size == 0:
                raise ValueError
            return chunk + 1

        self.func = func
        self.func_result = self.array + 1

    def test_non_lazy_input(self):
        # Check that a non-lazy input doesn't trip up the functionality.
        cube, cube_data = create_mock_cube(self.array)
        result = map_complete_blocks(
            cube, self.func, dims=(1,), out_sizes=(4,), dtype=self.array.dtype
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
            cube, self.func, dims=(1,), out_sizes=(4,), dtype=lazy_array.dtype
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), self.func_result)
        # check correct data was accessed
        cube.lazy_data.assert_called_once()
        cube_data.assert_not_called()

    def test_dask_array_input(self):
        lazy_array = da.asarray(self.array, chunks=((1, 1), (4,)))
        result = map_complete_blocks(
            lazy_array, self.func, dims=(1,), out_sizes=(4,), dtype=lazy_array.dtype
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), self.func_result)

    def test_dask_masked_array_input(self):
        array = da.ma.masked_array(np.arange(2), mask=np.arange(2))
        result = map_complete_blocks(
            array, self.func, dims=tuple(), out_sizes=tuple(), dtype=array.dtype
        )
        self.assertTrue(is_lazy_data(result))
        self.assertTrue(isinstance(da.utils.meta_from_array(result), np.ma.MaskedArray))
        self.assertArrayEqual(result.compute(), np.ma.masked_array([1, 2], mask=[0, 1]))

    def test_dask_array_input_with_different_output_dtype(self):
        lazy_array = da.ma.masked_array(self.array, chunks=((1, 1), (4,)))
        dtype = np.float32

        def func(chunk):
            if chunk.size == 0:
                raise ValueError
            return (chunk + 1).astype(np.float32)

        result = map_complete_blocks(
            lazy_array, func, dims=(1,), out_sizes=(4,), dtype=dtype
        )
        self.assertTrue(isinstance(da.utils.meta_from_array(result), np.ma.MaskedArray))
        self.assertTrue(result.dtype == dtype)
        self.assertTrue(result.compute().dtype == dtype)
        self.assertArrayEqual(result.compute(), self.func_result)

    def test_rechunk(self):
        lazy_array = da.asarray(self.array, chunks=((1, 1), (2, 2)))
        cube, _ = create_mock_cube(lazy_array)
        result = map_complete_blocks(
            cube, self.func, dims=(1,), out_sizes=(4,), dtype=lazy_array.dtype
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), self.func_result)

    def test_different_out_shape(self):
        lazy_array = da.asarray(self.array, chunks=((1, 1), (4,)))
        cube, _ = create_mock_cube(lazy_array)

        def func(_):
            return np.arange(2).reshape(1, 2)

        func_result = [[0, 1], [0, 1]]
        result = map_complete_blocks(
            cube, func, dims=(1,), out_sizes=(2,), dtype=lazy_array.dtype
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), func_result)

    def test_multidimensional_input(self):
        array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        lazy_array = da.asarray(array, chunks=((1, 1), (1, 2), (4,)))
        cube, _ = create_mock_cube(lazy_array)
        result = map_complete_blocks(
            cube, self.func, dims=(1, 2), out_sizes=(3, 4), dtype=lazy_array.dtype
        )
        self.assertTrue(is_lazy_data(result))
        self.assertArrayEqual(result.compute(), array + 1)


if __name__ == "__main__":
    tests.main()
