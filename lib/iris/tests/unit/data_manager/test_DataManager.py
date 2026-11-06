# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris._data_manager.DataManager`."""

import copy

import numpy as np
import numpy.ma as ma
import pytest

from iris._data_manager import DataManager
from iris._lazy_data import as_lazy_data
from iris.tests import _shared_utils


class Test__init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = np.array([1])

    def test_data_same_shape(self):
        msg = '"shape" should only be provided if "data" is None'
        with pytest.raises(ValueError, match=msg):
            DataManager(self.data, self.data.shape)

    def test_data_conflicting_shape(self):
        msg = '"shape" should only be provided if "data" is None'
        with pytest.raises(ValueError, match=msg):
            DataManager(self.data, ())

    def test_no_data_no_shape(self):
        msg = 'one of "shape" or "data" should be provided; both are None'
        with pytest.raises(ValueError, match=msg):
            DataManager(None, None)


class Test___copy__:
    def test(self):
        dm = DataManager(np.array(0))
        emsg = "Shallow-copy of {!r} is not permitted."
        name = type(dm).__name__
        with pytest.raises(copy.Error, match=emsg.format(name)):
            copy.copy(dm)


class Test___deepcopy__:
    def test(self, mocker):
        dm = DataManager(np.array(0))
        method = "iris._data_manager.DataManager._deepcopy"
        return_value = mocker.sentinel.return_value
        mocked = mocker.patch(method)
        mocked.return_value = return_value
        result = copy.deepcopy(dm)
        assert mocked.call_count == 1
        [args], kwargs = mocked.call_args
        assert kwargs == dict()
        assert len(args) == 2
        expected = [return_value, [dm]]
        for item in args.values():
            assert item in expected
        assert result is return_value


class Test___eq__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.shape = (2, 3, 4)
        self.size = np.prod(self.shape)
        self.real_array = np.arange(self.size, dtype=float).reshape(self.shape)

    def test_real_with_real(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.copy())
        assert dm1 == dm2

    def test_real_with_real_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(np.ones(self.shape))
        assert not dm1 == dm2

    def test_real_with_real__dtype_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.astype(int))
        assert not dm1 == dm2

    def test_real_with_lazy_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(as_lazy_data(self.real_array))
        assert not dm1 == dm2
        assert not dm2 == dm1

    def test_lazy_with_lazy(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array))
        assert dm1 == dm2

    def test_lazy_with_lazy_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array) * 10)
        assert not dm1 == dm2

    def test_lazy_with_lazy__dtype_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array).astype(int))
        assert not dm1 == dm2

    def test_dataless(self):
        dm1 = DataManager(data=None, shape=(1,))
        dm2 = DataManager(data=None, shape=(1,))
        assert dm1 == dm2

    def test_dataless_failure(self):
        dm1 = DataManager(data=None, shape=(1,))
        dm2 = DataManager(data=None, shape=(2,))
        assert dm1 != dm2

    def test_dataless_with_real(self):
        dm1 = DataManager(data=None, shape=(1,))
        dm2 = DataManager(self.real_array)
        assert not dm1 == dm2

    def test_non_data_manager_failure(self):
        dm = DataManager(np.array(0))
        assert not dm == 0


class Test___ne__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.shape = (2, 3, 4)
        self.size = np.prod(self.shape)
        self.real_array = np.arange(self.size, dtype=float).reshape(self.shape)

    def test_real_with_real(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(np.ones(self.shape))
        assert dm1 != dm2

    def test_real_with_real_failure(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.copy())
        assert not dm1 != dm2

    def test_real_with_real__dtype(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(self.real_array.astype(int))
        assert dm1 != dm2

    def test_real_with_lazy(self):
        dm1 = DataManager(self.real_array)
        dm2 = DataManager(as_lazy_data(self.real_array))
        assert dm1 != dm2
        assert dm2 != dm1

    def test_lazy_with_lazy(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array) * 10)
        assert dm1 != dm2

    def test_lazy_with_lazy_failure(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array))
        assert not dm1 != dm2

    def test_lazy_with_lazy__dtype(self):
        dm1 = DataManager(as_lazy_data(self.real_array))
        dm2 = DataManager(as_lazy_data(self.real_array).astype(int))
        assert dm1 != dm2

    def test_non_data_manager(self):
        dm = DataManager(np.array(0))
        assert dm != 0


class Test___repr__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.real_array = np.array(123)
        masked_array = ma.array([0, 1], mask=[0, 1])
        self.lazy_array = as_lazy_data(masked_array)
        self.name = DataManager.__name__

    def test_real(self):
        dm = DataManager(self.real_array)
        result = repr(dm)
        expected = "{}({!r})".format(self.name, self.real_array)
        assert result == expected

    def test_lazy(self):
        dm = DataManager(self.lazy_array)
        result = repr(dm)
        expected = "{}({!r})".format(self.name, self.lazy_array)
        assert result == expected

    def test_dataless(self):
        dm = DataManager(None, self.real_array.shape)
        result = repr(dm)
        expected = "{}({!r}), shape={}".format(self.name, None, self.real_array.shape)
        assert result == expected


class Test__assert_axioms:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.real_array = np.array(0)
        self.lazy_array = as_lazy_data(self.real_array)
        self.dm = DataManager(self.real_array)

    def test_array_none(self):
        self.dm._real_array = None
        self.dm._shape = None
        emsg = "Unexpected data state, got no lazy or real data, and no shape."
        with pytest.raises(ValueError, match=emsg):
            self.dm._assert_axioms()

    def test_array_all(self):
        self.dm._lazy_array = self.lazy_array
        emsg = "Unexpected data state, got both lazy and real data."
        with pytest.raises(ValueError, match=emsg):
            self.dm._assert_axioms()


class Test__deepcopy:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.shape = (2, 3, 4)
        self.size = np.prod(self.shape)
        self.real_array = np.arange(self.size, dtype=float).reshape(self.shape)
        self.memo = dict()

    def test_real(self):
        dm = DataManager(self.real_array)
        result = dm._deepcopy(self.memo)
        assert dm == result

    def test_lazy(self):
        dm = DataManager(as_lazy_data(self.real_array))
        result = dm._deepcopy(self.memo)
        assert dm == result

    def test_real_with_real(self):
        dm = DataManager(self.real_array)
        data = self.real_array.copy() * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        assert result == expected
        assert result._real_array is data

    def test_real_with_lazy(self):
        dm = DataManager(self.real_array)
        data = as_lazy_data(self.real_array) * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        assert result == expected
        assert result._lazy_array is data

    def test_lazy_with_real(self):
        dm = DataManager(as_lazy_data(self.real_array))
        data = self.real_array.copy() * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        assert result == expected
        assert result._real_array is data

    def test_lazy_with_lazy(self):
        dm = DataManager(as_lazy_data(self.real_array))
        data = as_lazy_data(self.real_array) * 10
        result = dm._deepcopy(self.memo, data=data)
        expected = DataManager(data)
        assert result == expected
        assert result._lazy_array is data

    def test_real_with_real_failure(self):
        dm = DataManager(self.real_array)
        emsg = "Cannot copy"
        with pytest.raises(ValueError, match=emsg):
            dm._deepcopy(self.memo, data=np.array(0))

    def test_real_with_lazy_failure(self):
        dm = DataManager(self.real_array)
        emsg = "Cannot copy"
        with pytest.raises(ValueError, match=emsg):
            dm._deepcopy(self.memo, data=as_lazy_data(np.array(0)))

    def test_lazy_with_real_failure(self):
        dm = DataManager(as_lazy_data(self.real_array))
        emsg = "Cannot copy"
        with pytest.raises(ValueError, match=emsg):
            dm._deepcopy(self.memo, data=np.array(0))

    def test_lazy_with_lazy_failure(self):
        dm = DataManager(as_lazy_data(self.real_array))
        emsg = "Cannot copy"
        with pytest.raises(ValueError, match=emsg):
            dm._deepcopy(self.memo, data=as_lazy_data(np.array(0)))


class Test_data__getter:
    @pytest.fixture(autouse=True)
    def _setup(self):
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
        assert not dm.has_lazy_data()
        result = dm.data
        assert not dm.has_lazy_data()
        assert result is self.real_array

    def test_with_lazy_array(self):
        dm = DataManager(self.lazy_array)
        assert dm.has_lazy_data()
        result = dm.data
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(result, self.lazy_array)
        _shared_utils.assert_array_equal(result, self.real_array)

    def test_with_lazy_mask_array__not_masked(self):
        dm = DataManager(self.lazy_mask_array)
        assert dm.has_lazy_data()
        result = dm.data
        assert not dm.has_lazy_data()
        assert isinstance(result, np.core.ndarray)
        assert dm.dtype == self.dtype
        assert result.fill_value == self.fill_value
        _shared_utils.assert_array_equal(result, self.real_array)

    def test_with_lazy_mask_array__masked(self):
        dm = DataManager(self.lazy_mask_array_masked)
        assert dm.has_lazy_data()
        result = dm.data
        assert not dm.has_lazy_data()
        assert isinstance(result, ma.MaskedArray)
        assert dm.dtype == self.dtype
        assert result.fill_value == self.fill_value
        _shared_utils.assert_array_equal(result, self.mask_array_masked)

    def test_with_real_masked_constant(self):
        masked_data = ma.masked_array([666], mask=True, dtype=np.dtype("f8"))
        masked_constant = masked_data[0]
        dm = DataManager(masked_constant)
        result = dm.data
        assert not dm.has_lazy_data()
        assert isinstance(result, ma.MaskedArray)
        assert not isinstance(result, ma.core.MaskedConstant)
        _shared_utils.assert_masked_array_equal(result, masked_data)

    def test_with_lazy_masked_constant(self):
        masked_data = ma.masked_array([666], mask=True)
        masked_constant = masked_data[0]
        lazy_masked_constant = as_lazy_data(masked_constant)
        dm = DataManager(lazy_masked_constant)
        result = dm.data
        assert not dm.has_lazy_data()
        assert isinstance(result, ma.MaskedArray)
        assert not isinstance(result, ma.core.MaskedConstant)
        _shared_utils.assert_masked_array_equal(result, masked_data)


class Test_data__setter:
    def test_zero_ndim_real_with_scalar_int(self):
        value = 456
        dm = DataManager(np.array(123))
        assert not dm.has_lazy_data()
        dm.data = value
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, np.array(value))

    def test_zero_ndim_real_with_scalar_float(self):
        value = 456.0
        dm = DataManager(np.array(123))
        assert not dm.has_lazy_data()
        dm.data = value
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, np.array(value))

    def test_zero_ndim_real_with_zero_ndim_real(self):
        real_array = np.array(456)
        dm = DataManager(np.array(123))
        assert not dm.has_lazy_data()
        dm.data = real_array
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_zero_ndim_real_with_zero_ndim_lazy(self):
        lazy_array = as_lazy_data(np.array(456))
        dm = DataManager(np.array(123))
        assert not dm.has_lazy_data()
        dm.data = lazy_array
        assert dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, lazy_array.compute())

    def test_zero_ndim_lazy_with_zero_ndim_real(self):
        real_array = np.array(456)
        dm = DataManager(as_lazy_data(np.array(123)))
        assert dm.has_lazy_data()
        dm.data = real_array
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_zero_ndim_lazy_with_zero_ndim_lazy(self):
        lazy_array = as_lazy_data(np.array(456))
        dm = DataManager(as_lazy_data(np.array(123)))
        assert dm.has_lazy_data()
        dm.data = lazy_array
        assert dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, lazy_array.compute())

    def test_zero_ndim_real_to_scalar_1d_real_promote(self):
        real_array = np.array([456])
        dm = DataManager(np.array(123))
        assert not dm.has_lazy_data()
        dm.data = real_array
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_zero_ndim_real_to_scalar_1d_lazy_promote(self):
        lazy_array = as_lazy_data(np.array([456]))
        dm = DataManager(np.array(123))
        assert not dm.has_lazy_data()
        dm.data = lazy_array
        assert dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, lazy_array.compute())

    def test_zero_ndim_lazy_to_scalar_1d_real_promote(self):
        real_array = np.array([456])
        dm = DataManager(as_lazy_data(np.array(123)))
        assert dm.has_lazy_data()
        dm.data = real_array
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_zero_ndim_lazy_to_scalar_1d_lazy_promote(self):
        lazy_array = as_lazy_data(np.array([456]))
        dm = DataManager(as_lazy_data(np.array(123)))
        assert dm.has_lazy_data()
        dm.data = lazy_array
        assert dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, lazy_array.compute())

    def test_scalar_1d_to_zero_ndim_fail(self):
        dm = DataManager(np.array([123]))
        emsg = r"Require data with shape \(1,\), got \(\)."
        with pytest.raises(ValueError, match=emsg):
            dm.data = 456

    def test_nd_real_to_nd_real(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        dm = DataManager(real_array * 10)
        assert not dm.has_lazy_data()
        dm.data = real_array
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_nd_real_to_nd_lazy(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array) * 10
        dm = DataManager(real_array)
        assert not dm.has_lazy_data()
        dm.data = lazy_array
        assert dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, lazy_array.compute())

    def test_nd_lazy_to_nd_real(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array * 10)
        assert dm.has_lazy_data()
        dm.data = real_array
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_nd_lazy_to_nd_lazy(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array * 10)
        assert dm.has_lazy_data()
        dm.data = lazy_array
        assert dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, lazy_array.compute())

    def test_nd_lazy_to_dataless(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array * 10)
        assert dm.has_lazy_data()
        dm.data = None
        assert dm.core_data() is None

    def test_nd_real_to_dataless(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        dm = DataManager(real_array)
        assert not dm.has_lazy_data()
        dm.data = None
        assert dm.core_data() is None

    def test_dataless_to_nd_lazy(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(None, shape)
        assert dm.shape == shape
        dm.data = lazy_array
        assert dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, lazy_array.compute())

    def test_dataless_to_nd_real(self):
        shape = (2, 3, 4)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        dm = DataManager(None, shape)
        assert dm.core_data() is None
        dm.data = real_array
        assert not dm.has_lazy_data()
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_coerce_to_ndarray(self):
        shape = (2, 3)
        size = np.prod(shape)
        real_array = np.arange(size).reshape(shape)
        matrix = np.matrix(real_array)
        dm = DataManager(real_array)
        dm.data = matrix
        assert isinstance(dm._real_array, np.core.ndarray)
        assert isinstance(dm.data, np.core.ndarray)
        _shared_utils.assert_array_equal(dm.data, real_array)

    def test_real_masked_constant_to_array(self):
        masked_data = ma.masked_array([666], mask=True, dtype=np.dtype("f8"))
        masked_constant = masked_data[0]
        dm = DataManager(masked_constant)
        assert isinstance(dm._real_array, ma.MaskedArray)
        assert not isinstance(dm._real_array, ma.core.MaskedConstant)
        assert isinstance(dm.data, ma.MaskedArray)
        assert not isinstance(dm.data, ma.core.MaskedConstant)
        _shared_utils.assert_masked_array_equal(dm.data, masked_data)


class Test_dtype:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.real_array = np.array(0, dtype=np.dtype("int64"))
        self.lazy_array = as_lazy_data(np.array(0, dtype=np.dtype("float64")))

    def test_real_array(self):
        dm = DataManager(self.real_array)
        assert dm.dtype == np.dtype("int64")

    def test_lazy_array(self):
        dm = DataManager(self.lazy_array)
        assert dm.dtype == np.dtype("float64")


class Test_ndim:
    def test_ndim_0(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        assert dm.ndim == 0
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        assert dm.ndim == 0

    def test_ndim_nd(self):
        shape = (2, 3, 4)
        real_array = np.arange(24).reshape(shape)
        dm = DataManager(real_array)
        assert dm.ndim == 3
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        assert dm.ndim == 3


class Test_shape:
    def test_shape_scalar(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        assert dm.shape == ()
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        assert dm.shape == ()

    def test_shape_nd(self):
        shape = (2, 3, 4)
        real_array = np.arange(24).reshape(shape)
        dm = DataManager(real_array)
        assert dm.shape == shape
        lazy_array = as_lazy_data(real_array)
        dm = DataManager(lazy_array)
        assert dm.shape == shape

    def test_shape_data_to_dataless(self):
        shape = (2, 3, 4)
        real_array = np.arange(24).reshape(shape)
        dm = DataManager(None, shape)
        assert dm.shape == shape
        dm.data = real_array
        assert dm.shape == shape

    def test_shape_dataless_to_data(self):
        shape = (2, 3, 4)
        real_array = np.arange(24).reshape(shape)
        dm = DataManager(real_array)
        assert dm.shape == shape
        dm.data = None
        assert dm.shape == shape


class Test_copy:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.method = "iris._data_manager.DataManager._deepcopy"
        self.data = mocker.sentinel.data
        self.return_value = mocker.sentinel.return_value
        self.memo = {}

    def test(self, mocker):
        dm = DataManager(np.array(0))
        kwargs = dict(data=self.data)
        mocked = mocker.patch(self.method)
        mocked.return_value = self.return_value
        result = dm.copy(data=self.data)
        mocked.assert_called_once_with(self.memo, **kwargs)
        assert result is self.return_value


class Test_core_data:
    def test_real_array(self):
        real_array = np.array(0)
        dm = DataManager(real_array)
        assert dm.core_data() is real_array

    def test_lazy_array(self):
        lazy_array = as_lazy_data(np.array(0))
        dm = DataManager(lazy_array)
        assert dm.core_data() is lazy_array


class Test_has_lazy_data:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.real_array = np.array(0)
        self.lazy_array = as_lazy_data(self.real_array)

    def test_with_lazy_array(self):
        dm = DataManager(self.lazy_array)
        assert dm.has_lazy_data()

    def test_with_real_array(self):
        dm = DataManager(self.real_array)
        assert not dm.has_lazy_data()


class Test_lazy_data:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.real_array = np.array(0)
        self.lazy_array = as_lazy_data(self.real_array)

    def test_with_real_array(self):
        dm = DataManager(self.real_array)
        assert not dm.has_lazy_data()
        result = dm.lazy_data()
        assert not dm.has_lazy_data()
        assert result == self.lazy_array
        assert not dm.has_lazy_data()

    def test_with_lazy_array(self):
        dm = DataManager(self.lazy_array)
        assert dm.has_lazy_data()
        result = dm.lazy_data()
        assert dm.has_lazy_data()
        assert result is dm._lazy_array


class Test_is_dataless:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = np.array(0)
        self.shape = (0,)

    def test_with_data(self):
        dm = DataManager(self.data)
        assert not dm.is_dataless()

    def test_without_data(self):
        dm = DataManager(None, self.shape)
        assert dm.is_dataless()
