# (C) British Crown Copyright 2019, Met Office
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
"""Unit tests for the `iris.fileformats.netcdf._get_cf_var_data` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from dask.array import Array as dask_array
import numpy as np

from iris._lazy_data import _optimum_chunksize
import iris.fileformats.cf
from iris.fileformats.netcdf import _get_cf_var_data
from iris.tests import mock


class Test__get_cf_var_data(tests.IrisTest):
    def setUp(self):
        self.filename = 'DUMMY'
        self.shape = (300000, 240, 200)
        self.expected_chunks = _optimum_chunksize(self.shape, self.shape)

    def _make(self, chunksizes):
        cf_data = mock.Mock(_FillValue=None)
        cf_data.chunking = mock.MagicMock(return_value=chunksizes)
        cf_var = mock.MagicMock(spec=iris.fileformats.cf.CFVariable,
                                dtype=np.dtype('i4'),
                                cf_data=cf_data,
                                cf_name='DUMMY_VAR',
                                shape=self.shape)
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
        chunks = 'contiguous'
        cf_var = self._make(chunks)
        lazy_data = _get_cf_var_data(cf_var, self.filename)
        lazy_data_chunks = [c[0] for c in lazy_data.chunks]
        self.assertArrayEqual(lazy_data_chunks, self.expected_chunks)


if __name__ == "__main__":
    tests.main()
