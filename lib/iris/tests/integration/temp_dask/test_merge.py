# (C) British Crown Copyright 2017, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Test lazy data utility functions.

Note: really belongs in "tests/unit/lazy_data".

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests


import numpy as np
import dask.array as da

import iris
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris._lazy_data import is_lazy_data
import iris._merge


def sample_lazy_data(shape):
    array = np.arange(1.0 * np.prod(shape)).reshape(shape)
    lazy_array = da.from_array(array, 1e6)
    return lazy_array


def sample_cube(shape=(2, 3), name='cube', units='1',
                x=None, y=None, z=None, **attrs):
    cube = Cube(sample_lazy_data(shape), units=units)
    cube.rename(name)
    cube.attributes.update(attrs)
    if x is None and len(shape) > 0:
        x = np.arange(shape[-1])
    if y is None and len(shape) > 1:
        y = np.arange(shape[-2])
    if x is not None:
        co = DimCoord(np.array(x, dtype=float), long_name='x', units='1')
        cube.add_dim_coord(co, len(shape) - 1)
    if y is not None:
        co = DimCoord(np.array(y, dtype=float), long_name='y')
        if len(shape) > 1:
            cube.add_dim_coord(co, len(shape) - 2)
        else:
            cube.add_aux_coord(co)
    if z is not None:
        co = DimCoord(np.array(z, dtype=float), long_name='z')
        if len(shape) > 2:
            cube.add_dim_coord(co, len(shape) - 3)
        else:
            cube.add_aux_coord(co)
    return cube


class TestMergeData(tests.IrisTest):
    def test_single_lazy(self):
        cube = sample_cube()
        self.assertTrue(cube.has_lazy_data())
        cubelist = CubeList([cube])
        merged = cubelist.merge()
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].has_lazy_data())
        self.assertTrue(cube.has_lazy_data())
        self.assertEqual(merged[0], cube)

    def test_single_concrete(self):
        cube = sample_cube()
        cube.data
        cubelist = CubeList([cube])
        merged = cubelist.merge()
        self.assertEqual(len(merged), 1)
        self.assertFalse(merged[0].has_lazy_data())
        self.assertEqual(merged[0], cube)

    def test_multiple_distinct(self):
        cubelist = CubeList([sample_cube(name='a1'),
                             sample_cube(name='a2')])
        merged = cubelist.merge()
        self.assertEqual(len(merged), 2)
        self.assertTrue(merged[0].has_lazy_data())
        self.assertTrue(merged[1].has_lazy_data())
        self.assertEqual(merged, cubelist)

    def _sample_merge_cubelist(self):
        cube1 = sample_cube(z=5)
        cube2 = sample_cube(z=7)
        cube2._my_data = cube2._my_data + 100.0  # NB different but still lazy
        cubelist = CubeList([cube1, cube2])
        return cubelist

    def _check_sample_merged_result(self, merged):
        cube, = merged
        self.assertArrayAlmostEqual(cube.coord('z').points, [5.0, 7.0])
        self.assertArrayAlmostEqual(cube.data,
                                    [[[0, 1, 2],
                                      [3, 4, 5]],
                                     [[100, 101, 102],
                                      [103, 104, 105]]])
        self.assertFalse(cube.has_lazy_data())

    def test_multiple_joined_all_lazy(self):
        cubelist = self._sample_merge_cubelist()
        merged = cubelist.merge()
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].has_lazy_data())
        self._check_sample_merged_result(merged)

    def test_multiple_joined_all_concrete(self):
        cubelist = self._sample_merge_cubelist()
        [cube.data for cube in cubelist]
        merged = cubelist.merge()
        self.assertEqual(len(merged), 1)
        self.assertFalse(merged[0].has_lazy_data())
        self._check_sample_merged_result(merged)

    def test_multiple_joined_mixed(self):
        cubelist = self._sample_merge_cubelist()
        cubelist[0].data
        merged = cubelist.merge()
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].has_lazy_data())
        self._check_sample_merged_result(merged)

    def _sample_multidim_merge_cubelist(self):
        cubes = []
        for i_y in range(6):
            for i_x in range(4):
                cube = sample_cube(z=i_y)
                cube.add_aux_coord(DimCoord(i_x, long_name='aux'))
                cubes.append(cube)
        return CubeList(cubes)

    def test_multidim_merge(self):
        # Check an example that requires a multi-dimensional stack operation.
        cubelist = self._sample_multidim_merge_cubelist()
        merged = cubelist.merge()
        self.assertEqual(len(merged), 1)
        cube, = merged
        self.assertTrue(cube.has_lazy_data())
        self.assertEqual(cube.shape, (4, 6, 2, 3))

    def test_multidim_merge__inner_call(self):
        # Do that again, to make sure that an nd-array is passed.
        cubelist = self._sample_multidim_merge_cubelist()

        # Patch the inner call in _merge, to record call arguments.
        original_dasktack_call = iris._merge._multidim_daskstack
        global _global_call_args
        _global_call_args = []

        def passthrough_daskstack_call(stack):
            global _global_call_args
            _global_call_args.append(stack)
            return original_dasktack_call(stack)

        stack_patch = self.patch('iris._merge._multidim_daskstack',
                                 passthrough_daskstack_call)

        # Do the merge.
        cubelist.merge()

        # Check the call sequence + what was passed.
        self.assertEqual([arg.shape for arg in _global_call_args],
                         [(4, 6), (6,), (6,), (6,), (6,)])
        last_arg = _global_call_args[-1]
        self.assertIsInstance(last_arg, np.ndarray)
        object_dtype = np.zeros((), dtype=object).dtype
        self.assertEqual(last_arg.dtype, object_dtype)
        self.assertTrue(is_lazy_data(last_arg[0]))


if __name__ == '__main__':
    tests.main()
