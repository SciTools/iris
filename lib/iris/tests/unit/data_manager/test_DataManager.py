# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris._data_manager.DataManager`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import copy
from unittest import mock

import numpy as np
import numpy.ma as ma

from iris._data_manager import DataManager
from iris._lazy_data import as_lazy_data


class Test___copy__(tests.IrisTest):
    def test(self):
        dm = DataManager(np.array(0))
        emsg = "Shallow-copy of {!r} is not permitted."
        name = type(dm).__name__
        with self.assertRaisesRegex(copy.Error, emsg.format(name)):
            copy.copy(dm)


class Test___deepcopy__(tests.IrisTest):
    def test(self):
        dm = DataManager(np.array(0))
        method = "iris._data_manager.DataManager._deepcopy"
        return_value = mock.sentinel.return_value
        with mock.patch(method) as mocker:
            mocker.return_value = return_value
            result = copy.deepcopy(dm)
            self.assertEqual(mocker.call_count, 1)
            [args], kwargs = mocker.call_args
            self.assertEqual(kwargs, dict())
            self.assertEqual(len(args), 2)
            expected = [return_value, [dm]]
            for item in args.values():
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

    def test_lazy_with_lazy__dtype_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array).astype(int))
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

    def test_lazy_with_lazy__dtype(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array).astype(int))
        self.assertNotEqual(dm1, dm2)

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
        expected = "{}({!r})".format(self.name, self.real_array)
        self.assertEqual(result, expected)

    def test_lazy(self):
        dm = DataManager(self.lazy_array)
        result = repr(dm)
        expected = "{}({!r})".format(self.name, self.lazy_array)
        self.assertEqual(result, expected)


class Test__assert_axioms(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0)
        self.lazy_array = as_lazy_data(self.real_array)
        self.dm = DataManager(self.real_array)

    def test_array_none(self):
        self.dm._real_array = None
        emsg = "Unexpected data state, got no lazy and no real data"
        with self.assertRaisesRegex(AssertionError, emsg):
            self.dm._assert_axioms()

    def test_array_all(self):
        self.dm._lazy_array = self.lazy_array
        emsg = "Unexpected data state, got lazy and real data"
        with self.assertRaisesRegex(AssertionError, emsg):
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

    def test_real_with_real_failure(self):
        dm = DataManager(self.real_array)
        emsg = "Cannot copy"
        with self.assertRaisesRegex(ValueError, emsg):
            dm._deepcopy(self.memo, data=np.array(0))

    def test_real_with_lazy_failure(self):
        dm = DataManager(self.real_array)
        emsg = "Cannot copy"
        with self.assertRaisesRegex(ValueError, emsg):
            dm._deepcopy(self.memo, data=as_lazy_data(np.array(0)))

    def test_lazy_with_real_failure(self):
        dm = DataManager(as_lazy_data(self.real_array))
        emsg = "Cannot copy"
        with self.assertRaisesRegex(ValueError, emsg):
            dm._deepcopy(self.memo, data=np.array(0))

    def test_lazy_with_lazy_failure(self):
        dm = DataManager(as_lazy_data(self.real_array))
        emsg = "Cannot copy"
        with self.assertRaisesRegex(ValueError, emsg):
            dm._deepcopy(self.memo, data=as_lazy_data(np.array(0)))


class Test_data__getter(tests.IrisTest):
    def setUp(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        self.real_array = np.arange(size).reshape(shape)
        self.lazy_array = as_lazy_data(self.real_array)
        self.mask_array = ma.masked_array(self.real_array)
        self.mask_array_masked = self.mask_array.copy()
        self.mask_array_masked[0, 0, 0] = ma.masked
        self.dtype = self.mask_array.dtype
        self.fill_value = self.mask_array.fill_value
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

    def test_with_lazy_mask_array__not_masked(self):
        dm = DataManager(self.lazy_mask_array)
        self.assertTrue(dm.has_lazy_data())
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIsInstance(result, np.core.ndarray)
        self.assertEqual(dm.dtype, self.dtype)
        self.assertEqual(result.fill_value, self.fill_value)
        self.assertArrayEqual(result, self.real_array)

    def test_with_lazy_mask_array__masked(self):
        dm = DataManager(self.lazy_mask_array_masked)
        self.assertTrue(dm.has_lazy_data())
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertEqual(dm.dtype, self.dtype)
        self.assertEqual(result.fill_value, self.fill_value)
        self.assertArrayEqual(result, self.mask_array_masked)

    def test_with_real_masked_constant(self):
        masked_data = ma.masked_array([666], mask=True, dtype=np.dtype("f8"))
        masked_constant = masked_data[0]
        dm = DataManager(masked_constant)
        result = dm.data
        self.assertFalse(dm.has_lazy_data())
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertNotIsInstance(result, ma.core.MaskedConstant)
        self.assertMaskedArrayEqual(result, masked_data)

    def test_with_lazy_masked_constant(self):
        masked_data = ma.masked_array([666], mask=True)
        masked_constant = masked_data[0]
        lazy_masked_constant = as_lazy_data(masked_constant)
        dm = DataManager(lazy_masked_constant)
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
        emsg = r"Require data with shape \(1,\), got \(\)."
        with self.assertRaisesRegex(ValueError, emsg):
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
        masked_data = ma.masked_array([666], mask=True, dtype=np.dtype("f8"))
        masked_constant = masked_data[0]
        dm = DataManager(masked_constant)
        self.assertIsInstance(dm._real_array, ma.MaskedArray)
        self.assertNotIsInstance(dm._real_array, ma.core.MaskedConstant)
        self.assertIsInstance(dm.data, ma.MaskedArray)
        self.assertNotIsInstance(dm.data, ma.core.MaskedConstant)
        self.assertMaskedArrayEqual(dm.data, masked_data)


class Test_dtype(tests.IrisTest):
    def setUp(self):
        self.real_array = np.array(0, dtype=np.dtype("int64"))
        self.lazy_array = as_lazy_data(np.array(0, dtype=np.dtype("float64")))

    def test_real_array(self):
        dm = DataManager(self.real_array)
        self.assertEqual(dm.dtype, np.dtype("int64"))

    def test_lazy_array(self):
        dm = DataManager(self.lazy_array)
        self.assertEqual(dm.dtype, np.dtype("float64"))


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
        self.method = "iris._data_manager.DataManager._deepcopy"
        self.data = mock.sentinel.data
        self.return_value = mock.sentinel.return_value
        self.memo = {}

    def test(self):
        dm = DataManager(np.array(0))
        kwargs = dict(data=self.data)
        with mock.patch(self.method) as mocker:
            mocker.return_value = self.return_value
            result = dm.copy(data=self.data)
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


if __name__ == "__main__":
    tests.main()
