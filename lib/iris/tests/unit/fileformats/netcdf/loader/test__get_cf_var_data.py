# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.netcdf._get_cf_var_data` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from dask.array import Array as dask_array
import numpy as np

from iris._lazy_data import _optimum_chunksize
import iris.fileformats.cf
from iris.fileformats.netcdf.loader import _get_cf_var_data


class Test__get_cf_var_data(tests.IrisTest):
    def setUp(self):
        self.filename = "DUMMY"
        self.shape = (300000, 240, 200)
        self.expected_chunks = _optimum_chunksize(self.shape, self.shape)

    def _make(self, chunksizes):
        cf_data = mock.Mock(_FillValue=None)
        cf_data.chunking = mock.MagicMock(return_value=chunksizes)
        cf_var = mock.MagicMock(
            spec=iris.fileformats.cf.CFVariable,
            dtype=np.dtype("i4"),
            cf_data=cf_data,
            cf_name="DUMMY_VAR",
            shape=self.shape,
        )
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


if __name__ == "__main__":
    tests.main()
