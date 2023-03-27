# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Tests for :meth:`iris.fileformats.netcdf.saver.Saver._lazy_stream_data`.
"""

from unittest import mock

import dask.array as da
import numpy as np
import pytest

import iris.fileformats.netcdf._thread_safe_nc as threadsafe_nc
from iris.fileformats.netcdf.saver import Saver


class Test_streaming:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path_factory):
        self.saver = Saver(
            filename=tmp_path_factory.mktemp("stream") / "tmp.nc",
            netcdf_format="NETCDF4",
        )
        self.fill_value = 2.0
        self.real_data = np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])
        self.lazy_data = da.from_array(self.real_data)
        self.cf_var = mock.MagicMock(
            spec=threadsafe_nc.VariableWrapper, shape=(3,), dtype=np.float32
        )
        self.emulated_data = mock.Mock(shape=(3,))
        self.emulated_var = mock.Mock(
            spec=threadsafe_nc.VariableWrapper,
            shape=(3,),
            dtype=np.float32,
            _in_memory_data=self.emulated_data,
        )

    def test_real_data(self):
        # When source is real data, it should be directly assigned to cf_var.
        self.saver._lazy_stream_data(
            data=self.real_data, cf_var=self.cf_var, fill_value=self.fill_value
        )
        assert self.cf_var.__setitem__.call_count == 1
        call_args = self.cf_var.__setitem__.call_args[0]
        assert call_args[0] == slice(None)
        assert (
            call_args[1] is self.real_data
        )  # N.B. equality here is *not* good enough !

    def test_lazy_data(self):
        # When source is lazy data, it should be passed to da.store.
        with mock.patch(
            "iris.fileformats.netcdf.saver.da.store"
        ) as mock_store:
            self.saver._lazy_stream_data(
                data=self.lazy_data,
                cf_var=self.cf_var,
                fill_value=self.fill_value,
            )
        assert mock_store.call_count == 1
        (arg1,) = mock_store.call_args[0][0]
        assert arg1 is self.lazy_data

    def test_emulated_data(self):
        # When the var is an "emulated" var, data should be directly assigned to its
        # '_in_memory_data' property.
        assert self.emulated_var._in_memory_data is self.emulated_data
        mock_newdata = mock.Mock(shape=(3,))
        self.saver._lazy_stream_data(
            data=mock_newdata,
            cf_var=self.emulated_var,
            fill_value=self.fill_value,
        )
        assert self.emulated_var._in_memory_data is mock_newdata
