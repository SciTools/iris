# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.netcdf._get_cf_var_data` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from dask.array import Array as dask_array
import numpy as np

from iris._lazy_data import _optimum_chunksize
import iris.fileformats.cf
from iris.fileformats.netcdf.loader import CHUNK_CONTROL, _get_cf_var_data


class Test__get_cf_var_data(tests.IrisTest):
    def setUp(self):
        self.filename = "DUMMY"
        self.shape = (300000, 240, 200)
        self.expected_chunks = _optimum_chunksize(self.shape, self.shape)

    def _make(self, chunksizes=None, shape=None, dtype="i4", **extra_properties):
        cf_data = mock.MagicMock(
            _FillValue=None,
            __getitem__="<real-data>",
            dimensions=["dim_" + str(x) for x in range(len(shape or "1"))],
        )
        cf_data.chunking = mock.MagicMock(return_value=chunksizes)
        if shape is None:
            shape = self.shape
        dtype = np.dtype(dtype)
        cf_var = mock.MagicMock(
            spec=iris.fileformats.cf.CFVariable,
            dtype=dtype,
            cf_data=cf_data,
            cf_name="DUMMY_VAR",
            shape=shape,
            size=np.prod(shape),
            **extra_properties,
        )
        cf_var.__getitem__.return_value = mock.sentinel.real_data_accessed
        return cf_var

    def test_cf_data_type(self):
        chunks = [1, 12, 100]
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var, self.filename)
        self.assertIsInstance(lazy_data, dask_array)

    def test_cf_data_chunks(self):
        chunks = [2500, 240, 200]
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var, self.filename)
        lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        expected_chunks = _optimum_chunksize(chunks, self.shape)
        self.assertArrayEqual(lazy_data_chunks, expected_chunks)

    def test_cf_data_chunk_control(self):
        # more thorough testing can be found at `test__chunk_control`
        chunks = [2500, 240, 200]
        cf_var = self._make(shape=(2500, 240, 200), chunksizes=chunks)
        with CHUNK_CONTROL.set(dim_0=25, dim_1=24, dim_2=20):
            lazy_data = _get_cf_var_data(cf_var, self.filename)
            lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        expected_chunks = (25, 24, 20)
        self.assertArrayEqual(lazy_data_chunks, expected_chunks)

    def test_cf_data_no_chunks(self):
        # No chunks means chunks are calculated from the array's shape by
        # `iris._lazy_data._optimum_chunksize()`.
        chunks = None
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var, self.filename)
        lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        self.assertArrayEqual(lazy_data_chunks, self.expected_chunks)

    def test_cf_data_contiguous(self):
        # Chunks 'contiguous' is equivalent to no chunks.
        chunks = "contiguous"
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var, self.filename)
        lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        self.assertArrayEqual(lazy_data_chunks, self.expected_chunks)

    def test_type__1kf8_is_lazy(self):
        cf_var = self._make(shape=(1000,), dtype="f8")
        var_data = _get_cf_var_data(cf_var, self.filename)
        self.assertIsInstance(var_data, dask_array)

    def test_arraytype__1ki2_is_real(self):
        cf_var = self._make(shape=(1000,), dtype="i2")
        var_data = _get_cf_var_data(cf_var, self.filename)
        self.assertIs(var_data, mock.sentinel.real_data_accessed)

    def test_arraytype__100f8_is_real(self):
        cf_var = self._make(shape=(100,), dtype="f8")
        var_data = _get_cf_var_data(cf_var, self.filename)
        self.assertIs(var_data, mock.sentinel.real_data_accessed)

    def test_cf_data_emulation(self):
        # Check that a variable emulation object passes its real data directly.
        emulated_data = mock.Mock()
        # Make a cf_var with a special extra '_data_array' property.
        cf_var = self._make(chunksizes=None, _data_array=emulated_data)
        result = _get_cf_var_data(cf_var, self.filename)
        # This should get directly returned.
        self.assertIs(emulated_data, result)


if __name__ == "__main__":
    tests.main()
