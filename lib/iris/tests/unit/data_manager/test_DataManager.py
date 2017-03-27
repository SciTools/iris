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
Unit tests for the :class:`iris._data_manager.DataManager`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._data_manager import DataManager


class Test__assert_axioms(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0)
        self.lazy_array = da.from_array(self.real_array, chunks=1)
        self.dm = DataManager(self.real_array)

    def test_data_none(self):
        self.dm._real_array = None
        emsg = 'Unexpected data state, got no lazy and no real data'
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_data_all(self):
        self.dm._lazy_array = self.lazy_array
        emsg = 'Unexpected data state, got lazy and real data'
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_realised_dtype(self):
        self.dm._realised_dtype = np.dtype('float')
        emsg = 'Unexpected realised dtype state, got dtype'
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_real_data_with_realised_dtype(self):
        self.dm._realised_dtype = np.dtype('int')
        emsg = ("Unexpected real data with realised dtype, got "
                "real data and realised dtype\('int64'\)")
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()


class Test__dtype_setter(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0)
        self.lazy_array = da.from_array(self.real_array, chunks=1)
        self.dm = DataManager(self.lazy_array)

    def test_with_none(self):
        self.assertIsNone(self.dm._realised_dtype)
        self.dm._dtype_setter(None)
        self.assertIsNone(self.dm._realised_dtype)

    def test_with_real_array(self):
        self.dm._lazy_array = None
        self.dm._real_array = self.real_array
        emsg = 'Cannot set realised dtype, no lazy data is available'
        with self.assertRaisesRegexp(ValueError, emsg):
            self.dm._dtype_setter(np.dtype('int'))

    def test_realised_dtype_bad(self):
        emsg = ("Can only cast lazy data to an integer or boolean "
                "dtype, got dtype\('float64'\)")
        with self.assertRaisesRegexp(ValueError, emsg):
            self.dm._dtype_setter(np.dtype('float64'))

    def test_realised_dtype(self):
        dtypes = (np.dtype('bool'), np.dtype('int64'), np.dtype('uint64'))
        for dtype in dtypes:
            self.dm._realised_dtype = None
            self.dm._dtype_setter(dtype)
            self.assertEqual(self.dm._realised_dtype, dtype)


class Test_core_data(tests.IrisTest):
    def test_real_array(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        self.assertIs(dm.core_data, real_array)

    def test_lazy_array(self):
        lazy_array = da.from_array(np.array(0), chunks=1)
        dm = DataManager(lazy_array)
        self.assertIs(dm.core_data, lazy_array)


class Test_dtype(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0, dtype=np.dtype('float64'))
        self.lazy_array = da.from_array(np.array(0, dtype=np.dtype('int64')),
                                                 chunks=1)
        
    def test_real_array(self):
        dm = DataManager(self.real_array)
        self.assertEqual(dm.dtype, np.dtype('float64'))

    def test_lazy_array(self):
        dm = DataManager(self.lazy_array)
        self.assertEqual(dm.dtype, np.dtype('int64'))

    def test_lazy_array_realised_dtype(self):
        dm = DataManager(self.lazy_array, realised_dtype=np.dtype('bool'))
        self.assertEqual(dm.dtype, np.dtype('bool'))
        self.assertEqual(dm._lazy_array.dtype, np.dtype('int64'))


class Test_ndim(tests.IrisTest):
    def test_ndim_0(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        self.assertEqual(dm.ndim, 0)
        lazy_array = da.from_array(real_array, chunks=1)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.ndim, 0)

    def test_ndim_nd(self):
        real_array = np.arange(24).reshape(2, 3, 4)
        dm = DataManager(real_array)
        self.assertEqual(dm.ndim, 3)
        lazy_array = da.from_array(real_array, chunks=1)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.ndim, 3)


class Test_shape(tests.IrisTest):
    def test_shape_scalar(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        self.assertEqual(dm.shape, ())
        lazy_array = da.from_array(real_array, chunks=1)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.shape, ())

    def test_shape_nd(self):
        shape = (2, 3, 4)
        real_array = np.arange(24).reshape(shape)
        dm = DataManager(real_array)
        self.assertEqual(dm.shape, shape)
        lazy_array = da.from_array(real_array, chunks=1)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.shape, shape)


class Test_has_lazy_data(tests.IrisTest):
    pass
        

if __name__ == '__main__':
    tests.main()
