# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.netcdf._get_cf_var_data` function."""

import dask.array as da
import numpy as np
import pytest

from iris._lazy_data import _optimum_chunksize
import iris.fileformats.cf
from iris.fileformats.netcdf._thread_safe_nc import VLType
from iris.fileformats.netcdf.loader import CHUNK_CONTROL, _get_cf_var_data
from iris.tests import _shared_utils
from iris.tests.unit.fileformats import MockerMixin


class Test__get_cf_var_data(MockerMixin):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.filename = "DUMMY"
        self.shape = (300000, 240, 200)
        self.expected_chunks = _optimum_chunksize(self.shape, self.shape)

    def _make(self, chunksizes=None, shape=None, dtype="i4", **extra_properties):
        cf_data = self.mocker.MagicMock(
            _FillValue=None,
            __getitem__="<real-data>",
            dimensions=["dim_" + str(x) for x in range(len(shape or "1"))],
        )
        cf_data.chunking = self.mocker.MagicMock(return_value=chunksizes)
        if shape is None:
            shape = self.shape
        if dtype is not str:  # for testing VLen str arrays (dtype=`class <str>`)
            dtype = np.dtype(dtype)
        cf_var = self.mocker.MagicMock(
            spec=iris.fileformats.cf.CFVariable,
            dtype=dtype,
            cf_data=cf_data,
            filename=self.filename,
            cf_name="DUMMY_VAR",
            shape=shape,
            size=np.prod(shape),
            **extra_properties,
        )
        cf_var.__getitem__.return_value = self.mocker.sentinel.real_data_accessed
        return cf_var

    def test_cf_data_type(self):
        chunks = [1, 12, 100]
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var)
        assert isinstance(lazy_data, da.Array)
        assert isinstance(da.utils.meta_from_array(lazy_data), np.ma.MaskedArray)

    def test_cf_data_chunks(self):
        chunks = [2500, 240, 200]
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var)
        lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        expected_chunks = _optimum_chunksize(chunks, self.shape)
        _shared_utils.assert_array_equal(lazy_data_chunks, expected_chunks)

    def test_cf_data_chunk_control(self):
        # more thorough testing can be found at `test__chunk_control`
        chunks = [2500, 240, 200]
        cf_var = self._make(shape=(2500, 240, 200), chunksizes=chunks)
        with CHUNK_CONTROL.set(dim_0=25, dim_1=24, dim_2=20):
            lazy_data = _get_cf_var_data(cf_var)
            lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        expected_chunks = (25, 24, 20)
        _shared_utils.assert_array_equal(lazy_data_chunks, expected_chunks)

    def test_cf_data_no_chunks(self):
        # No chunks means chunks are calculated from the array's shape by
        # `iris._lazy_data._optimum_chunksize()`.
        chunks = None
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var)
        lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        _shared_utils.assert_array_equal(lazy_data_chunks, self.expected_chunks)

    def test_cf_data_contiguous(self):
        # Chunks 'contiguous' is equivalent to no chunks.
        chunks = "contiguous"
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var)
        lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        _shared_utils.assert_array_equal(lazy_data_chunks, self.expected_chunks)

    def test_type__1kf8_is_lazy(self):
        cf_var = self._make(shape=(1000,), dtype="f8")
        var_data = _get_cf_var_data(cf_var)
        assert isinstance(var_data, da.Array)

    def test_arraytype__1ki2_is_real(self, mocker):
        cf_var = self._make(shape=(1000,), dtype="i2")
        var_data = _get_cf_var_data(cf_var)
        assert var_data is mocker.sentinel.real_data_accessed

    def test_arraytype__100f8_is_real(self, mocker):
        cf_var = self._make(shape=(100,), dtype="f8")
        var_data = _get_cf_var_data(cf_var)
        assert var_data is mocker.sentinel.real_data_accessed

    def test_vltype__1000str_is_lazy(self, mocker):
        # Variable length string type
        mock_vltype = mocker.Mock(spec=VLType, dtype=str, name="varlen string type")
        cf_var = self._make(shape=(1000,), dtype=str, datatype=mock_vltype)
        var_data = _get_cf_var_data(cf_var)
        assert isinstance(var_data, da.Array)

    def test_vltype__1000str_is_real_with_hint(self, mocker):
        # Variable length string type with a hint on the array variable length size
        mock_vltype = mocker.Mock(spec=VLType, dtype=str, name="varlen string type")
        cf_var = self._make(shape=(100,), dtype=str, datatype=mock_vltype)
        with CHUNK_CONTROL.set("DUMMY_VAR", _vl_hint=1):
            var_data = _get_cf_var_data(cf_var)
        assert var_data is mocker.sentinel.real_data_accessed

    def test_vltype__100str_is_real(self, mocker):
        # Variable length string type
        mock_vltype = mocker.Mock(spec=VLType, dtype=str, name="varlen string type")
        cf_var = self._make(shape=(100,), dtype=str, datatype=mock_vltype)
        var_data = _get_cf_var_data(cf_var)
        assert var_data is mocker.sentinel.real_data_accessed

    def test_vltype__100str_is_lazy_with_hint(self, mocker):
        # Variable length string type with a hint on the array variable length size
        mock_vltype = mocker.Mock(spec=VLType, dtype=str, name="varlen string type")
        cf_var = self._make(shape=(100,), dtype=str, datatype=mock_vltype)
        with CHUNK_CONTROL.set("DUMMY_VAR", _vl_hint=50):
            var_data = _get_cf_var_data(cf_var)
        assert isinstance(var_data, da.Array)

    def test_vltype__100f8_is_lazy(self, mocker):
        # Variable length float64 type
        mock_vltype = mocker.Mock(spec=VLType, dtype="f8", name="varlen float64 type")
        cf_var = self._make(shape=(1000,), dtype="f8", datatype=mock_vltype)
        var_data = _get_cf_var_data(cf_var)
        assert isinstance(var_data, da.Array)

    def test_vltype__100f8_is_real_with_hint(self, mocker):
        # Variable length float64 type with a hint on the array variable length size
        mock_vltype = mocker.Mock(spec=VLType, dtype="f8", name="varlen float64 type")
        cf_var = self._make(shape=(100,), dtype="f8", datatype=mock_vltype)
        with CHUNK_CONTROL.set("DUMMY_VAR", _vl_hint=2):
            var_data = _get_cf_var_data(cf_var)
        assert var_data is mocker.sentinel.real_data_accessed

    def test_cf_data_emulation(self, mocker):
        # Check that a variable emulation object passes its real data directly.
        emulated_data = mocker.Mock()
        # Make a cf_var with a special extra '_data_array' property.
        cf_var = self._make(chunksizes=None, _data_array=emulated_data)
        result = _get_cf_var_data(cf_var)
        # This should get directly returned.
        assert emulated_data is result
