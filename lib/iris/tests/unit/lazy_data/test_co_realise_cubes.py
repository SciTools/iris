# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._lazy data.co_realise_cubes`."""

import numpy as np

from iris._lazy_data import as_lazy_data, co_realise_cubes
from iris.cube import Cube
from iris.tests._shared_utils import assert_array_all_close


class ArrayAccessCounter:
    def __init__(self, array):
        self.dtype = array.dtype
        self.shape = array.shape
        self.ndim = array.ndim
        self._array = array
        self.access_count = 0
        self.meta = np.empty((0,) * array.ndim, dtype=array.dtype)

    def __getitem__(self, keys):
        self.access_count += 1
        return self._array[keys]


class Test_co_realise_cubes:
    def test_empty(self):
        # Ensure that 'no args' case does not raise an error.
        co_realise_cubes()

    def test_basic(self):
        real_data = np.arange(3.0)
        cube = Cube(as_lazy_data(real_data))
        co_realise_cubes(cube)
        assert not cube.has_lazy_data()
        assert_array_all_close(cube.core_data(), real_data)

    def test_multi(self):
        real_data = np.arange(3.0)
        cube_base = Cube(as_lazy_data(real_data))
        cube_inner = cube_base + 1
        result_a = cube_base + 1
        result_b = cube_inner + 1
        co_realise_cubes(result_a, result_b)
        # Check that target cubes were realised.
        assert not result_a.has_lazy_data()
        assert not result_b.has_lazy_data()
        # Check that other cubes referenced remain lazy.
        assert cube_base.has_lazy_data()
        assert cube_inner.has_lazy_data()

    def test_combined_access(self):
        wrapped_array = ArrayAccessCounter(np.arange(3.0))
        lazy_array = as_lazy_data(wrapped_array, meta=wrapped_array.meta)
        derived_a = lazy_array + 1
        derived_b = lazy_array + 2
        derived_c = lazy_array + 3
        derived_d = lazy_array + 4
        derived_e = lazy_array + 5
        cube_a = Cube(derived_a)
        cube_b = Cube(derived_b)
        cube_c = Cube(derived_c)
        cube_d = Cube(derived_d)
        cube_e = Cube(derived_e)
        co_realise_cubes(cube_a, cube_b, cube_c, cube_d, cube_e)
        # Though used more than once, the source data should only get fetched
        # once by dask, when the whole data is accessed.
        # This also ensures that dask does *not* perform an initial data
        # access with no data payload to ascertain the metadata associated with
        # the dask.array (this access is specific to dask 2+,
        # see dask.array.utils.meta_from_array).
        assert wrapped_array.access_count == 1
