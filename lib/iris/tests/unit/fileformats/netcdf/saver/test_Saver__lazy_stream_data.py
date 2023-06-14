# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :meth:`iris.fileformats.netcdf.saver.Saver._lazy_stream_data`.

The behaviour of this method is complex, and this only tests certain aspects.
The testing of the dask delayed operations and file writing are instead covered by
integration tests.

"""
from unittest import mock
import warnings

import dask.array as da
import numpy as np
import pytest

import iris.fileformats.netcdf._thread_safe_nc as nc_threadsafe
from iris.fileformats.netcdf.saver import Saver, _FillvalueCheckInfo


class Test__lazy_stream_data:
    @staticmethod
    @pytest.fixture(autouse=True)
    def saver_patch():
        # Install patches, so we can create a Saver without opening a real output file.
        # Mock just enough of Dataset behaviour to allow a 'Saver.complete()' call.
        mock_dataset = mock.MagicMock()
        mock_dataset_class = mock.Mock(return_value=mock_dataset)
        # Mock the wrapper within the netcdf saver
        target1 = (
            "iris.fileformats.netcdf.saver._thread_safe_nc.DatasetWrapper"
        )
        # Mock the real netCDF4.Dataset within the threadsafe-nc module, as this is
        # used by NetCDFDataProxy and NetCDFWriteProxy.
        target2 = "iris.fileformats.netcdf._thread_safe_nc.netCDF4.Dataset"
        with mock.patch(target1, mock_dataset_class):
            with mock.patch(target2, mock_dataset_class):
                yield

    # A fixture to parametrise tests over delayed and non-delayed Saver type.
    # NOTE: this only affects the saver context-exit, which we do not test here, so
    # should make ***no difference to any of these tests***.
    @staticmethod
    @pytest.fixture(params=[False, True], ids=["nocompute", "compute"])
    def compute(request) -> bool:
        yield request.param

    # A fixture to parametrise tests over real and lazy-type data.
    @staticmethod
    @pytest.fixture(params=[False, True], ids=["realdata", "lazydata"])
    def data_is_lazy(request) -> bool:
        yield request.param

    @staticmethod
    def saver(compute) -> Saver:
        # Create a test Saver object
        return Saver(
            filename="<dummy>", netcdf_format="NETCDF4", compute=compute
        )

    @staticmethod
    def mock_var(shape):
        # Create a test cf_var object
        return mock.MagicMock(shape=tuple(shape), dtype=np.dtype(np.float32))

    def test_data_save(self, compute, data_is_lazy):
        """Real data is transferred immediately, lazy data creates a delayed write."""
        saver = self.saver(compute=compute)
        data = np.arange(5.0)
        if data_is_lazy:
            data = da.from_array(data)
        fill_value = -1.0  # not occurring in data
        cf_var = self.mock_var(data.shape)
        saver._lazy_stream_data(
            data=data, fill_value=fill_value, fill_warn=True, cf_var=cf_var
        )
        assert cf_var.__setitem__.call_count == (0 if data_is_lazy else 1)
        assert len(saver._delayed_writes) == (1 if data_is_lazy else 0)
        if data_is_lazy:
            result_data, result_writer, fill_info = saver._delayed_writes[0]
            assert result_data is data
            assert isinstance(result_writer, nc_threadsafe.NetCDFWriteProxy)
            assert isinstance(fill_info, _FillvalueCheckInfo)
        else:
            cf_var.__setitem__.assert_called_once_with(slice(None), data)

    def test_warnings(self, compute, data_is_lazy):
        """
        For real data, fill-value warnings are issued immediately.  For lazy data,
        warnings are returned from computing a delayed completion.

        N.B. The 'compute' keyword has **no effect** on this :  It only causes delayed
        writes to be automatically actioned on exiting a Saver context.
        Streaming *always* creates delayed writes for lazy data, since this is required
        to make dask distributed operation work.
        """
        saver = self.saver(compute=compute)
        data = np.arange(5.0)
        if data_is_lazy:
            data = da.from_array(data)
        fill_value = 2.0  # IS occurring in data
        cf_var = self.mock_var(data.shape)

        # Do initial save.  When compute=True, this issues warnings
        with warnings.catch_warnings(record=True) as logged_warnings:
            saver._lazy_stream_data(
                data=data, fill_value=fill_value, fill_warn=True, cf_var=cf_var
            )

        issued_warnings = [log.message for log in logged_warnings]

        n_expected_warnings = 0 if data_is_lazy else 1
        assert len(issued_warnings) == n_expected_warnings

        # Complete the write : any delayed warnings should be *returned*.
        # NOTE:
        #   (1) this still works when there are no delayed writes.
        #   (2) the Saver 'compute' keyword makes no difference to this usage, as it
        #       *only* affects what happens when the saver context exits.
        result2 = saver.delayed_completion().compute()
        issued_warnings += list(result2)

        # Either way, a suitable warning should have been produced.
        assert len(issued_warnings) == 1
        warning = issued_warnings[0]
        msg = "contains unmasked data points equal to the fill-value, 2.0"
        assert isinstance(warning, UserWarning)
        assert msg in warning.args[0]
