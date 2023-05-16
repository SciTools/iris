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

import iris.fileformats.netcdf._thread_safe_nc as threadsafe_nc
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
    @pytest.fixture(params=["realdata", "lazydata", "emulateddata"])
    def data_form(request) -> bool:
        yield request.param

    @staticmethod
    def saver(compute) -> Saver:
        # Create a test Saver object
        return Saver(
            filename="<dummy>", netcdf_format="NETCDF4", compute=compute
        )

    @staticmethod
    def mock_var(shape, with_data_array):
        # Create a test cf_var object.
        # N.B. using 'spec=' so we can control whether it has a '_data_array' property.
        if with_data_array:
            extra_properties = {
                "_data_array": mock.sentinel.initial_data_array
            }
        else:
            extra_properties = {}
        mock_cfvar = mock.MagicMock(
            spec=threadsafe_nc.VariableWrapper,
            shape=tuple(shape),
            dtype=np.dtype(np.float32),
            **extra_properties,
        )
        # Give the mock cf-var a name property, as required by '_lazy_stream_data'.
        # This *can't* be an extra kwarg to MagicMock __init__, since that already
        # defines a specific 'name' kwarg, with a different purpose.
        mock_cfvar.name = "<mock_cfvar>"
        return mock_cfvar

    def test_data_save(self, compute, data_form):
        """Real data is transferred immediately, lazy data creates a delayed write."""
        saver = self.saver(compute=compute)

        data = np.arange(5.0)
        if data_form == "lazydata":
            data = da.from_array(data)

        cf_var = self.mock_var(
            data.shape, with_data_array=(data_form == "emulateddata")
        )
        fill_value = -1.0  # not occurring in data
        saver._lazy_stream_data(
            data=data, fill_value=fill_value, fill_warn=True, cf_var=cf_var
        )
        if data_form == "lazydata":
            expect_n_setitem = 0
            expect_n_delayed = 1
        elif data_form == "realdata":
            expect_n_setitem = 1
            expect_n_delayed = 0
        else:
            assert data_form == "emulateddata"
            expect_n_setitem = 0
            expect_n_delayed = 0

        assert cf_var.__setitem__.call_count == expect_n_setitem
        assert len(saver._delayed_writes) == expect_n_delayed

        if data_form == "lazydata":
            result_data, result_writer, fill_info = saver._delayed_writes[0]
            assert result_data is data
            assert isinstance(result_writer, threadsafe_nc.NetCDFWriteProxy)
            assert isinstance(fill_info, _FillvalueCheckInfo)
        elif data_form == "realdata":
            cf_var.__setitem__.assert_called_once_with(slice(None), data)
        else:
            assert data_form == "emulateddata"
            cf_var._data_array == mock.sentinel.exact_data_array

    def test_warnings(self, compute, data_form):
        """
        For real data, fill-value warnings are issued immediately.
        For lazy data, warnings are returned from computing a delayed completion.
        For 'emulated' data (direct array transfer), no checks + no warnings ever.

        N.B. The 'compute' keyword has **no effect** on this :  It only causes delayed
        writes to be automatically actioned on exiting a Saver context.
        Streaming *always* creates delayed writes for lazy data, since this is required
        to make dask distributed operation work.
        """
        saver = self.saver(compute=compute)

        data = np.arange(5.0)
        if data_form == "lazydata":
            data = da.from_array(data)

        fill_value = 2.0  # IS occurring in data
        cf_var = self.mock_var(
            data.shape, with_data_array=(data_form == "emulateddata")
        )

        # Do initial save.  When compute=True, this issues warnings
        with warnings.catch_warnings(record=True) as logged_warnings:
            saver._lazy_stream_data(
                data=data, fill_value=fill_value, fill_warn=True, cf_var=cf_var
            )

        # Check warnings issued by initial call.
        issued_warnings = [log.message for log in logged_warnings]
        if data_form == "lazydata":
            n_expected_warnings = 0
        elif data_form == "realdata":
            n_expected_warnings = 1
        else:
            # No checks in the emulated case
            assert data_form == "emulateddata"
            n_expected_warnings = 0
        assert len(issued_warnings) == n_expected_warnings

        # Complete the write : any delayed warnings should be *returned*.
        # NOTE:
        #   (1) this still works when there are no delayed writes.
        #   (2) the Saver 'compute' keyword makes no difference to this usage, as it
        #       *only* affects what happens when the saver context exits.
        result2 = saver.delayed_completion().compute()
        issued_warnings += list(result2)

        # Check warnings issued during 'completion'.
        if data_form == "emulateddata":
            # No checks in this case, ever.
            n_expected_warnings = 0
        else:
            # Otherwise, either way, a suitable warning should now have been produced.
            n_expected_warnings = 1
        assert len(issued_warnings) == n_expected_warnings
        if n_expected_warnings > 0:
            warning = issued_warnings[0]
            msg = "contains unmasked data points equal to the fill-value, 2.0"
            assert isinstance(warning, UserWarning)
            assert msg in warning.args[0]
