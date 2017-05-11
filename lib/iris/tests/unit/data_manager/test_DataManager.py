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

import copy
import mock
import numpy as np
import numpy.ma as ma
import six

from iris._data_manager import DataManager
from iris._lazy_data import as_lazy_data


class Test___copy__(tests.IrisTest):
    def test(self):
        dm = DataManager(np.array(0))
        emsg = 'Shallow-copy of {!r} is not permitted.'
        name = type(dm).__name__
        with self.assertRaisesRegexp(copy.Error, emsg.format(name)):
            copy.copy(dm)


class Test___deepcopy__(tests.IrisTest):
    def test(self):
        dm = DataManager(np.array(0))
        method = 'iris._data_manager.DataManager._deepcopy'
        return_value = mock.sentinel.return_value
        with mock.patch(method) as mocker:
            mocker.return_value = return_value
            result = copy.deepcopy(dm)
            self.assertEqual(mocker.call_count, 1)
            [args], kwargs = mocker.call_args
            self.assertEqual(kwargs, dict())
            self.assertEqual(len(args), 2)
            expected = [return_value, [dm]]
            for item in six.itervalues(args):
                self.assertIn(item, expected)
        self.assertIs(result, return_value)


class Test___eq__(tests.IrisTest):
    def setUp(self):
        self.shape = (2, 3, 4)
        self.size = np.prod(self.shape)
        self.real_array = np.arange(self.size, dtype=float).reshape(self.shape)

    def test_real_with_real(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.copy())
        self.assertEqual(dm1, dm2)

    def test_real_with_real_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(np.ones(self.shape))
        self.assertFalse(dm1 == dm2)

    def test_real_with_real__fill_value(self):
        fill_value = 1234
        dm1 = DataManager(self.real_array, fill_value=fill_value)
        dm2 = DataManager(self.real_array, fill_value=fill_value)
        self.assertEqual(dm1, dm2)

    def test_real_with_real__fill_value_failure(self):
        dm1 = DataManager(self.real_array, fill_value=1234)
        dm2 = DataManager(self.real_array, fill_value=4321)
        self.assertFalse(dm1 == dm2)

    def test_real_with_real__dtype_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.astype(int))
        self.assertFalse(dm1 == dm2)

    def test_real_with_lazy_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(as_lazy_data(self.real_array))
        self.assertFalse(dm1 == dm2)
        self.assertFalse(dm2 == dm1)

    def test_lazy_with_lazy(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array))
        self.assertEqual(dm1, dm2)

    def test_lazy_with_lazy_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array) * 10)
        self.assertFalse(dm1 == dm2)

    def test_lazy_with_lazy__fill_value(self):
        fill_value = 1234
        dm1 = DataManager(as_lazy_data(self.real_array),
                          fill_value=fill_value)
        dm2 = DataManager(as_lazy_data(self.real_array),
                          fill_value=fill_value)
        self.assertEqual(dm1, dm2)

    def test_lazy_with_lazy__fill_value_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array),
                          fill_value=1234)
        dm2 = DataManager(as_lazy_data(self.real_array),
                          fill_value=4321)
        self.assertFalse(dm1 == dm2)

    def test_lazy_with_lazy__dtype_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array).astype(int))
        self.assertFalse(dm1 == dm2)

    def test_lazy_with_lazy__realised_dtype(self):
        dtype = np.dtype('int16')
        dm1 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=dtype)
        dm2 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=dtype)
        self.assertEqual(dm1, dm2)

    def test_lazy_with_lazy__realised_dtype_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=np.dtype('int8'))
        dm2 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=np.dtype('int16'))
        self.assertFalse(dm1 == dm2)

    def test_non_DataManager_failure(self):
        dm = DataManager(np.array(0))
        self.assertFalse(dm == 0)


class Test___ne__(tests.IrisTest):
    def setUp(self):
        self.shape = (2, 3, 4)
        self.size = np.prod(self.shape)
        self.real_array = np.arange(self.size, dtype=float).reshape(self.shape)

    def test_real_with_real(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(np.ones(self.shape))
        self.assertNotEqual(dm1, dm2)

    def test_real_with_real_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.copy())
        self.assertFalse(dm1 != dm2)

    def test_real_with_real__fill_value(self):
        dm1 = DataManager(self.real_array, fill_value=1234)
        dm2 = DataManager(self.real_array, fill_value=4321)
        self.assertNotEqual(dm1, dm2)

    def test_real_with_real__fill_value_failure(self):
        fill_value = 1234
        dm1 = DataManager(self.real_array, fill_value=fill_value)
        dm2 = DataManager(self.real_array, fill_value=fill_value)
        self.assertFalse(dm1 != dm2)

    def test_real_with_real__dtype(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.astype(int))
        self.assertNotEqual(dm1, dm2)

    def test_real_with_lazy(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(as_lazy_data(self.real_array))
        self.assertNotEqual(dm1, dm2)
        self.assertNotEqual(dm2, dm1)

    def test_lazy_with_lazy(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array) * 10)
        self.assertNotEqual(dm1, dm2)

    def test_lazy_with_lazy_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array))
        self.assertFalse(dm1 != dm2)

    def test_lazy_with_lazy__fill_value(self):
        dm1 = DataManager(as_lazy_data(self.real_array),
                          fill_value=1234)
        dm2 = DataManager(as_lazy_data(self.real_array),
                          fill_value=4321)
        self.assertNotEqual(dm1, dm2)

    def test_lazy_with_lazy__fill_value_failure(self):
        fill_value = 1234
        dm1 = DataManager(as_lazy_data(self.real_array),
                          fill_value=fill_value)
        dm2 = DataManager(as_lazy_data(self.real_array),
                          fill_value=fill_value)
        self.assertFalse(dm1 != dm2)

    def test_lazy_with_lazy__dtype(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array).astype(int))
        self.assertNotEqual(dm1, dm2)

    def test_lazy_with_lazy__realised_dtype(self):
        dm1 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=np.dtype('int8'))
        dm2 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=np.dtype('int16'))
        self.assertNotEqual(dm1, dm2)

    def test_lazy_with_lazy__realised_dtype_failure(self):
        dtype = np.dtype('int16')
        dm1 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=dtype)
        dm2 = DataManager(as_lazy_data(self.real_array),
                          realised_dtype=dtype)
        self.assertFalse(dm1 != dm2)

    def test_non_DataManager(self):
        dm = DataManager(np.array(0))
        self.assertNotEqual(dm, 0)


class Test___repr__(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(123)
        masked_array = ma.array([0, 1], mask=[0, 1])
        self.lazy_array = as_lazy_data(masked_array)
        self.name = DataManager.__name__

    def test_real(self):
        dm = DataManager(self.real_array)
        result = repr(dm)
        expected = '{}({!r})'.format(self.name, self.real_array)
        self.assertEqual(result, expected)

    def test_real__fill_value(self):
        fill_value = 1234
        dm = DataManager(self.real_array, fill_value=fill_value)
        result = repr(dm)
        fmt = '{}({!r}, fill_value={!r})'
        expected = fmt.format(self.name, self.real_array, fill_value)
        self.assertEqual(result, expected)

    def test_lazy(self):
        dm = DataManager(self.lazy_array)
        result = repr(dm)
        expected = '{}({!r})'.format(self.name, self.lazy_array)
        self.assertEqual(result, expected)

    def test_lazy__fill_value(self):
        fill_value = 1234.0
        dm = DataManager(self.lazy_array, fill_value=fill_value)
        result = repr(dm)
        fmt = '{}({!r}, fill_value={!r})'
        expected = fmt.format(self.name, self.lazy_array, fill_value)
        self.assertEqual(result, expected)

    def test_lazy__realised_dtype(self):
        dtype = np.dtype('int16')
        dm = DataManager(self.lazy_array, realised_dtype=dtype)
        result = repr(dm)
        fmt = '{}({!r}, realised_dtype={!r})'
        expected = fmt.format(self.name, self.lazy_array, dtype)
        self.assertEqual(result, expected)

    def test_lazy__fill_value_realised_dtype(self):
        fill_value = 1234
        dtype = np.dtype('int16')
        dm = DataManager(self.lazy_array, fill_value=fill_value,
                         realised_dtype=dtype)
        result = repr(dm)
        fmt = '{}({!r}, fill_value={!r}, realised_dtype={!r})'
        expected = fmt.format(self.name, self.lazy_array, fill_value, dtype)
        self.assertEqual(result, expected)


class Test__assert_axioms(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0)
        self.lazy_array = as_lazy_data(self.real_array)
        self.dm = DataManager(self.real_array)

    def test_array_none(self):
        self.dm._real_array = None
        emsg = 'Unexpected data state, got no lazy and no real data'
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_array_all(self):
        self.dm._lazy_array = self.lazy_array
        emsg = 'Unexpected data state, got lazy and real data'
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_realised_dtype(self):
        self.dm._realised_dtype = np.dtype('float')
        emsg = 'Unexpected realised dtype state, got dtype'
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_real_array_with_realised_dtype(self):
        self.dm._realised_dtype = np.dtype('int')
        emsg = ("Unexpected real data with realised dtype, got "
                "real data and realised dtype\('int64'\)")
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_non_float_lazy_array_with_realised_dtype(self):
        self.dm._real_array = None
        self.dm._lazy_array = self.lazy_array
        self.dm._realised_dtype = np.dtype('int')
        emsg = ("Unexpected lazy data dtype with realised dtype, got "
                "lazy data dtype\('int64'\) and realised "
                "dtype\('int64'\)")
        with self.assertRaisesRegexp(AssertionError, emsg):
            self.dm._assert_axioms()


class Test__deepcopy(tests.IrisTest):
    def setUp(self):
        self.shape = (2, 3, 4)
        self.size = np.prod(self.shape)
        self.real_array = np.arange(self.size, dtype=float).reshape(self.shape)
        self.memo = dict()

    def test_real(self):
        dm = DataManager(self.real_array)
        result = dm._deepcopy(self.memo)
        self.assertEqual(dm, result)

    def test_lazy(self):
        dm = DataManager(as_lazy_data(self.real_array))
        result = dm._deepcopy(self.memo)
        self.assertEqual(dm, result)

    def test_real_with_real(self):
        dm = DataManager(self.real_array)
        data = self.real_array.copy() * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        self.assertEqual(result, expected)
        self.assertIs(result._real_array, data)

    def test_real_with_lazy(self):
        dm = DataManager(self.real_array)
        data = as_lazy_data(self.real_array) * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        self.assertEqual(result, expected)
        self.assertIs(result._lazy_array, data)

    def test_lazy_with_real(self):
        dm = DataManager(as_lazy_data(self.real_array))
        data = self.real_array.copy() * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        self.assertEqual(result, expected)
        self.assertIs(result._real_array, data)

    def test_lazy_with_lazy(self):
        dm = DataManager(as_lazy_data(self.real_array))
        data = as_lazy_data(self.real_array) * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        self.assertEqual(result, expected)
        self.assertIs(result._lazy_array, data)

    def test_real_with_realised_dtype_failure(self):
        dm = DataManager(self.real_array)
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, realised_dtype=np.dtype('int16'))

    def test_lazy_with_realised_dtype__lazy_dtype_failure(self):
        dm = DataManager(as_lazy_data(self.real_array).astype(int))
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, realised_dtype=np.dtype('int16'))

    def test_lazy_with_realised_dtype_failure(self):
        dm = DataManager(as_lazy_data(self.real_array))
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, realised_dtype=np.dtype('float32'))

    def test_lazy_with_realised_dtype(self):
        dm = DataManager(as_lazy_data(self.real_array),
                         realised_dtype=np.dtype('int16'))
        data = as_lazy_data(self.real_array) * 10
        dtype = np.dtype('int8')
        result = dm._deepcopy(self.memo, data=data, realised_dtype=dtype)
        expected = DataManager(data, realised_dtype=dtype)
        self.assertEqual(result, expected)
        self.assertIs(result._lazy_array, data)

    def test_real_with_real_failure(self):
        dm = DataManager(self.real_array)
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, data=np.array(0))

    def test_real_with_lazy_failure(self):
        dm = DataManager(self.real_array)
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, data=as_lazy_data(np.array(0)))

    def test_lazy_with_real_failure(self):
        dm = DataManager(as_lazy_data(self.real_array))
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, data=np.array(0))

    def test_lazy_with_lazy_failure(self):
        dm = DataManager(as_lazy_data(self.real_array))
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, data=as_lazy_data(np.array(0)))

    def test__default_fill_value(self):
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)
        result = dm._deepcopy(self.memo)
        self.assertIsNone(result.fill_value)

    def test__default_with_fill_value(self):
        fill_value = 1234
        dm = DataManager(np.array(0), fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)
        result = dm._deepcopy(self.memo)
        self.assertEqual(result.fill_value, fill_value)

    def test__clear_fill_value(self):
        fill_value = 1234
        dm = DataManager(np.array(0), fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)
        result = dm._deepcopy(self.memo, fill_value=None)
        self.assertIsNone(result.fill_value)

    def test__set_fill_value(self):
        fill_value = 1234
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)
        result = dm._deepcopy(self.memo, fill_value=fill_value)
        self.assertEqual(result.fill_value, fill_value)

    def test__set_fill_value_failure(self):
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)
        emsg = 'Cannot copy'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm._deepcopy(self.memo, fill_value=1e+20)


class Test__realised_dtype_setter(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0.0)
        self.lazy_array = as_lazy_data(self.real_array)
        self.dm = DataManager(self.lazy_array)

    def test_lazy_with_none(self):
        self.assertIsNone(self.dm._realised_dtype)
        self.dm._realised_dtype_setter(None)
        self.assertIsNone(self.dm._realised_dtype)

    def test_real_with_none(self):
        self.dm._lazy_array = None
        self.dm._real_array = self.real_array
        self.assertIsNone(self.dm._realised_dtype)
        self.dm._realised_dtype_setter(None)
        self.assertIsNone(self.dm._realised_dtype)

    def test_real_with_same_dtype(self):
        self.dm._lazy_array = None
        self.dm._real_array = self.real_array
        self.assertIsNone(self.dm._realised_dtype)
        self.dm._realised_dtype_setter(self.dm.dtype)
        self.assertIsNone(self.dm._realised_dtype)

    def test_lazy_with_same_dtype(self):
        self.assertIsNone(self.dm._realised_dtype)
        self.dm._realised_dtype_setter(self.dm.dtype)
        self.assertIsNone(self.dm._realised_dtype)

    def test_real_array_failure(self):
        self.dm._lazy_array = None
        self.dm._real_array = self.real_array
        self.assertIsNone(self.dm._realised_dtype)
        emsg = 'Cannot set realised dtype, no lazy data is available'
        with self.assertRaisesRegexp(ValueError, emsg):
            self.dm._realised_dtype_setter(np.dtype('int16'))

    def test_invalid_realised_dtype(self):
        emsg = ("Can only cast lazy data to an integer or boolean "
                "dtype, got dtype\('float32'\)")
        with self.assertRaisesRegexp(ValueError, emsg):
            self.dm._realised_dtype_setter(np.dtype('float32'))

    def test_lazy_with_realised_dtype(self):
        dtypes = (np.dtype('bool'), np.dtype('int16'), np.dtype('uint16'))
        for dtype in dtypes:
            self.dm._realised_dtype = None
            self.dm._realised_dtype_setter(dtype)
            self.assertEqual(self.dm._realised_dtype, dtype)

    def test_lazy_with_realised_dtype__lazy_dtype_failure(self):
        self.dm._lazy_array = self.lazy_array.astype(np.dtype('int64'))
        emsg = ("Cannot set realised dtype for lazy data "
                "with dtype\('int64'\)")
        with self.assertRaisesRegexp(ValueError, emsg):
            self.dm._realised_dtype_setter(np.dtype('int16'))

    def test_lazy_replace_with_none(self):
        self.assertIsNone(self.dm._realised_dtype)
        dtype = np.dtype('int16')
        self.dm._realised_dtype_setter(dtype)
        self.assertEqual(self.dm._realised_dtype, dtype)
        self.dm._realised_dtype_setter(None)
        self.assertIsNone(self.dm._realised_dtype)


class Test_data__getter(tests.IrisTest):
    def setUp(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        self.real_array = np.arange(size).reshape(shape)
        self.lazy_array = as_lazy_data(self.real_array)
        self.mask_array = ma.masked_array(self.real_array)
        self.mask_array_masked = self.mask_array.copy()
        self.mask_array_masked[0, 0, 0] = ma.masked
        self.realised_dtype = self.mask_array.dtype
        self.lazy_mask_array = as_lazy_data(self.mask_array)
        self.lazy_mask_array_masked = as_lazy_data(self.mask_array_masked)

    def test_with_real_array(self):
        dm = DataManager(self.real_array)
        self.assertFalse(dm.has_lazy_data())
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIs(result, self.real_array)

    def test_with_lazy_array(self):
        dm = DataManager(self.lazy_array)
        self.assertTrue(dm.has_lazy_data())
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(result, self.real_array)

    def test_with_real_mask_array__default_fill_value(self):
        fill_value = 1234
        self.mask_array.fill_value = fill_value
        dm = DataManager(self.mask_array)
        self.assertEqual(dm.fill_value, fill_value)
        self.assertEqual(dm.data.fill_value, fill_value)

    def test_with_real_mask_array__with_fill_value_None(self):
        fill_value = 1234
        self.mask_array.fill_value = fill_value
        dm = DataManager(self.mask_array, fill_value=None)
        self.assertIsNone(dm.fill_value)
        np_fill_value = ma.array(0, dtype=dm.dtype).fill_value
        self.assertEqual(dm.data.fill_value, np_fill_value)

    def test_with_real_mask_array__with_fill_value(self):
        fill_value = 1234
        dm = DataManager(self.mask_array, fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)
        self.assertEqual(dm.data.fill_value, fill_value)

    def test_with_lazy_mask_array__masked_default_fill_value(self):
        dm = DataManager(self.lazy_mask_array_masked,
                         realised_dtype=self.realised_dtype)
        self.assertIsNone(dm.fill_value)
        np_fill_value = ma.array(0, dtype=dm.dtype).fill_value
        self.assertEqual(dm.data.fill_value, np_fill_value)

    def test_with_lazy_mask_array__masked_fill_value_None(self):
        dm = DataManager(self.lazy_mask_array_masked,
                         fill_value=None,
                         realised_dtype=self.realised_dtype)
        self.assertIsNone(dm.fill_value)
        np_fill_value = ma.array(0, dtype=dm.dtype).fill_value
        self.assertEqual(dm.data.fill_value, np_fill_value)

    def test_with_lazy_mask_array__masked_with_fill_value(self):
        fill_value = 1234
        dm = DataManager(self.lazy_mask_array_masked,
                         fill_value=fill_value,
                         realised_dtype=self.realised_dtype)
        self.assertEqual(dm.fill_value, fill_value)
        self.assertEqual(dm.data.fill_value, fill_value)

    def test_with_lazy_mask_array__not_masked(self):
        dm = DataManager(self.lazy_mask_array,
                         realised_dtype=self.realised_dtype)
        self.assertTrue(dm.has_lazy_data())
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIsInstance(result, np.core.ndarray)
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, self.realised_dtype)
        self.assertArrayEqual(result, self.real_array)

    def test_with_lazy_mask_array__masked(self):
        dm = DataManager(self.lazy_mask_array_masked,
                         realised_dtype=self.realised_dtype)
        self.assertTrue(dm.has_lazy_data())
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, self.realised_dtype)
        self.assertArrayEqual(result, self.lazy_mask_array_masked.compute())

    def test_with_real_masked_constant(self):
        masked_data = ma.masked_array([666], mask=True, dtype=np.dtype('f8'))
        masked_constant = masked_data[0]
        dm = DataManager(masked_constant)
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertNotIsInstance(result, ma.core.MaskedConstant)
        self.assertMaskedArrayEqual(result, masked_data)

    def test_with_lazy_masked_constant(self):
        dtype = np.dtype('int16')
        masked_data = ma.masked_array([666], mask=True, dtype=dtype)
        masked_constant = masked_data[0]
        lazy_masked_constant = as_lazy_data(masked_constant)
        dm = DataManager(lazy_masked_constant, realised_dtype=dtype)
        self.assertEqual(dm.dtype, dtype)
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertNotIsInstance(result, ma.core.MaskedConstant)
        self.assertMaskedArrayEqual(result, masked_data)


class Test_data__setter(tests.IrisTest):
    def test_zero_ndim_real_with_scalar_int(self):
        value = 456
        dm = DataManager(np.array(123))
        self.assertFalse(dm.has_lazy_data())
        dm.data = value
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, np.array(value))

    def test_zero_ndim_real_with_scalar_float(self):
        value = 456.0
        dm = DataManager(np.array(123))
        self.assertFalse(dm.has_lazy_data())
        dm.data = value
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, np.array(value))

    def test_zero_ndim_real_with_zero_ndim_real(self):
        real_array = np.array(456)
        dm = DataManager(np.array(123))
        self.assertFalse(dm.has_lazy_data())
        dm.data = real_array
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, real_array)

    def test_zero_ndim_real_with_zero_ndim_lazy(self):
        lazy_array = as_lazy_data(np.array(456))
        dm = DataManager(np.array(123))
        self.assertFalse(dm.has_lazy_data())
        dm.data = lazy_array
        self.assertTrue(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, lazy_array.compute())

    def test_zero_ndim_lazy_with_zero_ndim_real(self):
        real_array = np.array(456)
        dm = DataManager(as_lazy_data(np.array(123)))
        self.assertTrue(dm.has_lazy_data())
        dm.data = real_array
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, real_array)

    def test_zero_ndim_lazy_with_zero_ndim_lazy(self):
        lazy_array = as_lazy_data(np.array(456))
        dm = DataManager(as_lazy_data(np.array(123)))
        self.assertTrue(dm.has_lazy_data())
        dm.data = lazy_array
        self.assertTrue(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, lazy_array.compute())

    def test_zero_ndim_real_to_scalar_1d_real_promote(self):
        real_array = np.array([456])
        dm = DataManager(np.array(123))
        self.assertFalse(dm.has_lazy_data())
        dm.data = real_array
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, real_array)

    def test_zero_ndim_real_to_scalar_1d_lazy_promote(self):
        lazy_array = as_lazy_data(np.array([456]))
        dm = DataManager(np.array(123))
        self.assertFalse(dm.has_lazy_data())
        dm.data = lazy_array
        self.assertTrue(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, lazy_array.compute())

    def test_zero_ndim_lazy_to_scalar_1d_real_promote(self):
        real_array = np.array([456])
        dm = DataManager(as_lazy_data(np.array(123)))
        self.assertTrue(dm.has_lazy_data())
        dm.data = real_array
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, real_array)

    def test_zero_ndim_lazy_to_scalar_1d_lazy_promote(self):
        lazy_array = as_lazy_data(np.array([456]))
        dm = DataManager(as_lazy_data(np.array(123)))
        self.assertTrue(dm.has_lazy_data())
        dm.data = lazy_array
        self.assertTrue(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, lazy_array.compute())

    def test_scalar_1d_to_zero_ndim_fail(self):
        dm = DataManager(np.array([123]))
        emsg = 'Require data with shape \(1,\), got \(\).'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm.data = 456

    def test_nd_real_to_nd_real(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        dm = DataManager(real_array * 10)
        self.assertFalse(dm.has_lazy_data())
        dm.data = real_array
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, real_array)

    def test_nd_real_to_nd_lazy(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array) * 10
        dm = DataManager(real_array)
        self.assertFalse(dm.has_lazy_data())
        dm.data = lazy_array
        self.assertTrue(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, lazy_array.compute())

    def test_nd_lazy_to_nd_real(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array * 10)
        self.assertTrue(dm.has_lazy_data())
        dm.data = real_array
        self.assertFalse(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, real_array)

    def test_nd_lazy_to_nd_lazy(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array * 10)
        self.assertTrue(dm.has_lazy_data())
        dm.data = lazy_array
        self.assertTrue(dm.has_lazy_data())
        self.assertArrayEqual(dm.data, lazy_array.compute())

    def test_realised_dtype_none(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        mask_array = ma.arange(size).reshape(shape)
        dtype = mask_array.dtype
        lazy_array = as_lazy_data(mask_array)
        dm = DataManager(lazy_array, realised_dtype=dtype)
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)
        dm.data = mask_array
        self.assertIs(dm.data, mask_array)
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)

    def test_realised_dtype_clearance(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        mask_array = ma.arange(size).reshape(shape)
        mask_array[0, 0, 0] = ma.masked
        dtype = mask_array.dtype
        lazy_array = as_lazy_data(mask_array)
        dm = DataManager(lazy_array, realised_dtype=dtype)
        self.assertEqual(dm._realised_dtype, dtype)
        self.assertEqual(dm.dtype, dtype)
        dm.data = mask_array
        self.assertIs(dm.data, mask_array)
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)

    def test_coerce_to_ndarray(self):
        shape = (2, 3)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        matrix = np.matrix(real_array)
        dm = DataManager(real_array)
        dm.data = matrix
        self.assertIsInstance(dm._real_array, np.core.ndarray)
        self.assertIsInstance(dm.data, np.core.ndarray)
        self.assertArrayEqual(dm.data, real_array)

    def test_real_masked_constant_to_array(self):
        masked_data = ma.masked_array([666], mask=True, dtype=np.dtype('f8'))
        masked_constant = masked_data[0]
        dm = DataManager(masked_constant)
        self.assertIsInstance(dm._real_array, ma.MaskedArray)
        self.assertNotIsInstance(dm._real_array, ma.core.MaskedConstant)
        self.assertIsInstance(dm.data, ma.MaskedArray)
        self.assertNotIsInstance(dm.data, ma.core.MaskedConstant)
        self.assertMaskedArrayEqual(dm.data, masked_data)

    def test_real_array__fill_value_clearance(self):
        fill_value = 1234
        dm = DataManager(np.array(0), fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)
        dm.data = np.array(1)
        self.assertIsNone(dm.fill_value)

    def test_lazy_array__fill_value_clearance(self):
        fill_value = 1234
        dm = DataManager(as_lazy_data(np.array(0)),
                         fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)
        dm.data = as_lazy_data(np.array(1))
        self.assertIsNone(dm.fill_value)

    def test_mask_array__default_fill_value(self):
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)
        data = ma.array(1)
        dm.data = data
        self.assertIsNone(dm.fill_value)

    def test_mask_array__with_fill_value(self):
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)
        fill_value = 1234
        data = ma.array(1, fill_value=fill_value)
        dm.data = data
        self.assertEqual(dm.fill_value, fill_value)


class Test_dtype(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0, dtype=np.dtype('int64'))
        self.lazy_array = as_lazy_data(np.array(0, dtype=np.dtype('float64')))

    def test_real_array(self):
        dm = DataManager(self.real_array)
        self.assertEqual(dm.dtype, np.dtype('int64'))

    def test_lazy_array(self):
        dm = DataManager(self.lazy_array)
        self.assertEqual(dm.dtype, np.dtype('float64'))

    def test_lazy_array_realised_dtype(self):
        dm = DataManager(self.lazy_array, realised_dtype=np.dtype('bool'))
        self.assertEqual(dm.dtype, np.dtype('bool'))
        self.assertEqual(dm._lazy_array.dtype, np.dtype('float64'))


class Test_fill_value(tests.IrisTest):
    def setUp(self):
        self.dtypes = (np.dtype('int'), np.dtype('uint'),
                       np.dtype('bool'), np.dtype('float'))

    def test___init___default(self):
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)

    def test___init___with_fill_value(self):
        fill_value = 1234
        dm = DataManager(np.array(0), fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)

    def test_fill_value_bool(self):
        fill_value = np.bool_(True)
        for dtype in self.dtypes:
            data = np.array([0], dtype=dtype)
            dm = DataManager(data)
            dm.fill_value = fill_value
            [expected] = np.asarray([fill_value], dtype=dtype)
            self.assertEqual(dm.fill_value, expected)
            self.assertEqual(dm.fill_value.dtype, dtype)

    def test_fill_value_int(self):
        fill_value = np.int(1234)
        for dtype in self.dtypes:
            data = np.array([0], dtype=dtype)
            dm = DataManager(data)
            dm.fill_value = fill_value
            [expected] = np.asarray([fill_value], dtype=dtype)
            self.assertEqual(dm.fill_value, expected)
            self.assertEqual(dm.fill_value.dtype, dtype)

    def test_fill_value_float(self):
        fill_value = np.float(123.4)
        for dtype in self.dtypes:
            data = np.array([0], dtype=dtype)
            dm = DataManager(data)
            dm.fill_value = fill_value
            if dtype.kind in 'biu':
                fill_value = np.rint(fill_value)
            [expected] = np.asarray([fill_value], dtype=dtype)
            self.assertEqual(dm.fill_value, expected)
            self.assertEqual(dm.fill_value.dtype, dtype)

    def test_fill_value_uint(self):
        fill_value = np.uint(1234)
        for dtype in self.dtypes:
            data = np.array([0], dtype=dtype)
            dm = DataManager(data)
            dm.fill_value = fill_value
            [expected] = np.array([fill_value], dtype=dtype)
            self.assertEqual(dm.fill_value, expected)
            self.assertEqual(dm.fill_value.dtype, dtype)

    def test_fill_value_overflow(self):
        fill_value = np.float(1e+20)
        data = np.array([0], dtype=np.int32)
        dm = DataManager(data)
        emsg = 'Fill value of .* invalid for dtype'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm.fill_value = fill_value


class Test_ndim(tests.IrisTest):
    def test_ndim_0(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        self.assertEqual(dm.ndim, 0)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.ndim, 0)

    def test_ndim_nd(self):
        shape = (2, 3, 4)
        real_array = np.arange(24).reshape(shape)
        dm = DataManager(real_array)
        self.assertEqual(dm.ndim, len(shape))
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.ndim, len(shape))


class Test_shape(tests.IrisTest):
    def test_shape_scalar(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        self.assertEqual(dm.shape, ())
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.shape, ())

    def test_shape_nd(self):
        shape = (2, 3, 4)
        real_array = np.arange(24).reshape(shape)
        dm = DataManager(real_array)
        self.assertEqual(dm.shape, shape)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        self.assertEqual(dm.shape, shape)


class Test_copy(tests.IrisTest):
    def setUp(self):
        self.method = 'iris._data_manager.DataManager._deepcopy'
        self.data = mock.sentinel.data
        self.return_value = mock.sentinel.return_value
        self.memo = {}

    def test(self):
        dm = DataManager(np.array(0))
        kwargs = dict(data=self.data, fill_value='none', realised_dtype='none')
        with mock.patch(self.method) as mocker:
            mocker.return_value = self.return_value
            result = dm.copy(data=self.data)
            mocker.assert_called_once_with(self.memo, **kwargs)
        self.assertIs(result, self.return_value)

    def test_with_fill_value(self):
        dm = DataManager(np.array(0))
        fill_value = mock.sentinel.fill_value
        kwargs = dict(data=self.data, fill_value=fill_value,
                      realised_dtype='none')
        with mock.patch(self.method) as mocker:
            mocker.return_value = self.return_value
            result = dm.copy(data=self.data, fill_value=fill_value)
            mocker.assert_called_once_with(self.memo, **kwargs)
        self.assertIs(result, self.return_value)

    def test_with_realised_dtype(self):
        dm = DataManager(np.array(0))
        realised_dtype = mock.sentinel.realised_dtype
        kwargs = dict(data=self.data, fill_value='none',
                      realised_dtype=realised_dtype)
        with mock.patch(self.method) as mocker:
            mocker.return_value = self.return_value
            result = dm.copy(data=self.data, realised_dtype=realised_dtype)
            mocker.assert_called_once_with(self.memo, **kwargs)
        self.assertIs(result, self.return_value)


class Test_core_data(tests.IrisTest):
    def test_real_array(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        self.assertIs(dm.core_data(), real_array)

    def test_lazy_array(self):
        lazy_array = as_lazy_data(np.array(0))
        dm = DataManager(lazy_array)
        self.assertIs(dm.core_data(), lazy_array)


class Test_has_lazy_data(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0)
        self.lazy_array = as_lazy_data(self.real_array)

    def test_with_lazy_array(self):
        dm = DataManager(self.lazy_array)
        self.assertTrue(dm.has_lazy_data())

    def test_with_real_array(self):
        dm = DataManager(self.real_array)
        self.assertFalse(dm.has_lazy_data())


class Test_lazy_data(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0)
        self.lazy_array = as_lazy_data(self.real_array)

    def test_with_real_array(self):
        dm = DataManager(self.real_array)
        self.assertFalse(dm.has_lazy_data())
        result = dm.lazy_data()
        self.assertFalse(dm.has_lazy_data())
        self.assertEqual(result, self.lazy_array)
        self.assertFalse(dm.has_lazy_data())

    def test_with_lazy_array(self):
        dm = DataManager(self.lazy_array)
        self.assertTrue(dm.has_lazy_data())
        result = dm.lazy_data()
        self.assertTrue(dm.has_lazy_data())
        self.assertIs(result, dm._lazy_array)


class Test_replace(tests.IrisTest):
    def setUp(self):
        self.shape = (2, 3, 4)
        self.size = np.prod(self.shape)
        self.real_array = np.arange(self.size).reshape(self.shape)

    def test_real_with_real(self):
        dm = DataManager(self.real_array * 10)
        self.assertFalse(dm.has_lazy_data())
        dm.replace(self.real_array)
        self.assertFalse(dm.has_lazy_data())
        self.assertIs(dm._real_array, self.real_array)
        self.assertIs(dm.data, self.real_array)

    def test_real_with_lazy(self):
        lazy_array = as_lazy_data(self.real_array)
        dm = DataManager(self.real_array)
        self.assertFalse(dm.has_lazy_data())
        dm.replace(lazy_array)
        self.assertTrue(dm.has_lazy_data())
        self.assertIs(dm._lazy_array, lazy_array)
        self.assertArrayEqual(dm.data, self.real_array)

    def test_lazy_with_real(self):
        lazy_array = as_lazy_data(self.real_array)
        dm = DataManager(lazy_array * 10)
        self.assertTrue(dm.has_lazy_data())
        dm.replace(self.real_array)
        self.assertFalse(dm.has_lazy_data())
        self.assertIs(dm._real_array, self.real_array)
        self.assertIs(dm.data, self.real_array)

    def test_lazy_with_lazy(self):
        lazy_array = as_lazy_data(self.real_array)
        dm = DataManager(lazy_array * 10)
        self.assertTrue(dm.has_lazy_data())
        dm.replace(lazy_array)
        self.assertTrue(dm.has_lazy_data())
        self.assertIs(dm._lazy_array, lazy_array)
        self.assertArrayEqual(dm.data, self.real_array)

    def test_lazy_with_real__realised_dtype_none(self):
        mask_array = ma.arange(self.size).reshape(self.shape)
        dtype = mask_array.dtype
        lazy_array = as_lazy_data(mask_array)
        dm = DataManager(lazy_array, realised_dtype=dtype)
        self.assertTrue(dm.has_lazy_data())
        self.assertIsNone(dm._realised_dtype, dtype)
        self.assertEqual(dm.dtype, dtype)
        dm.replace(mask_array)
        self.assertFalse(dm.has_lazy_data())
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)
        self.assertIs(dm._real_array, mask_array)
        self.assertIs(dm.data, mask_array)

    def test_lazy_with_real__realised_dtype_clearance(self):
        mask_array = ma.arange(self.size).reshape(self.shape)
        mask_array[0, 0, 0] = ma.masked
        dtype = mask_array.dtype
        lazy_array = as_lazy_data(mask_array)
        dm = DataManager(lazy_array, realised_dtype=dtype)
        self.assertTrue(dm.has_lazy_data())
        self.assertEqual(dm._realised_dtype, dtype)
        self.assertEqual(dm.dtype, dtype)
        dm.replace(mask_array)
        self.assertFalse(dm.has_lazy_data())
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)
        self.assertIs(dm._real_array, mask_array)
        self.assertIs(dm.data, mask_array)

    def test_real_with_lazy__realised_dtype_setter_none(self):
        mask_array = ma.masked_array(self.real_array) * 10
        dtype = mask_array.dtype
        lazy_array = as_lazy_data(mask_array)
        dm = DataManager(self.real_array)
        self.assertFalse(dm.has_lazy_data())
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)
        dm.replace(lazy_array, realised_dtype=dtype)
        self.assertTrue(dm.has_lazy_data())
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)
        self.assertIs(dm._lazy_array, lazy_array)
        self.assertArrayEqual(dm.data, mask_array)

    def test_real_with_lazy__realised_dtype_setter(self):
        mask_array = ma.masked_array(self.real_array) * 10
        mask_array[0, 0, 0] = ma.masked
        dtype = mask_array.dtype
        lazy_array = as_lazy_data(mask_array)
        lazy_dtype = np.dtype('int16')
        dm = DataManager(self.real_array)
        self.assertFalse(dm.has_lazy_data())
        self.assertIsNone(dm._realised_dtype)
        self.assertEqual(dm.dtype, dtype)
        dm.replace(lazy_array, realised_dtype=lazy_dtype)
        self.assertTrue(dm.has_lazy_data())
        self.assertEqual(dm._realised_dtype, lazy_dtype)
        self.assertEqual(dm.dtype, lazy_dtype)
        self.assertIs(dm._lazy_array, lazy_array)
        self.assertArrayEqual(dm.data, mask_array)

    def test_real_with_real__realised_dtype_failure(self):
        dm = DataManager(self.real_array)
        emsg = 'Cannot set realised dtype, no lazy data is available.'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm.replace(self.real_array * 10,
                       realised_dtype=np.dtype('int16'))
        self.assertIs(dm._real_array, self.real_array)
        self.assertArrayEqual(dm.data, self.real_array)

    def test_real_with_real__promote_shape_with_dtype_failure(self):
        data = np.array(666)
        dm = DataManager(data)
        emsg = 'Cannot set realised dtype, no lazy data is available'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm.replace(np.array([999]),
                       realised_dtype=np.dtype('float32'))
        self.assertArrayEqual(dm.data, data)

    def test__clear_fill_value(self):
        fill_value = 1234
        dm = DataManager(np.array(0), fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)
        dm.replace(np.array(1))
        self.assertIsNone(dm.fill_value)

    def test__clear_fill_value_masked(self):
        fill_value = 1234
        dm = DataManager(np.array(0), fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)
        data = ma.masked_array(1, fill_value=4321)
        dm.replace(data)
        self.assertIsNone(dm.fill_value)

    def test__set_fill_value(self):
        fill_value = 1234
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)
        dm.replace(np.array(0), fill_value=fill_value)
        self.assertEqual(dm.fill_value, fill_value)

    def test__set_fill_value_failure(self):
        dm = DataManager(np.array(0))
        self.assertIsNone(dm.fill_value)
        emsg = 'Fill value of .* invalid for dtype'
        with self.assertRaisesRegexp(ValueError, emsg):
            dm.replace(np.array(1), fill_value=1e+20)


if __name__ == '__main__':
    tests.main()
