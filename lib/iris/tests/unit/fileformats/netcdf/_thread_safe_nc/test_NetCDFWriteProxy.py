# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._thread_safe_nc.NetCDFWriteProxy`."""

from threading import Lock

import netCDF4 as nc
from netCDF4 import Dataset as DatasetOriginal
import pytest

from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper, NetCDFWriteProxy


@pytest.fixture
def dataset_path(tmp_path):
    return tmp_path / "test.nc"


@pytest.fixture
def netcdf_variable(dataset_path):
    dataset = DatasetWrapper(dataset_path, "w")
    _ = dataset.createDimension("dim1", 1)
    variable = dataset.createVariable(
        "test_var",
        "f4",
        ("dim1",),
    )
    return variable


@pytest.fixture
def write_proxy(netcdf_variable) -> NetCDFWriteProxy:
    dataset = netcdf_variable.group()
    proxy = NetCDFWriteProxy(
        filepath=dataset.filepath(),
        cf_var=netcdf_variable,
        file_write_lock=Lock(),
    )
    dataset.close()
    return proxy


class UnreliableDatasetMaker:
    """A mock operation that returns a Dataset, but fails the first time it is called.

    This simulates non-deterministic HDF locking errors which are difficult to
    debug at the Python layer - pending further investigation.
    """

    def __init__(self, attempts_before_success=1):
        self.attempts_before_success = attempts_before_success
        self.call_count = 0

    def __call__(self, *args, **kwargs) -> nc.Dataset:
        self.call_count += 1
        if self.call_count <= self.attempts_before_success:
            raise OSError("Simulated non-deterministic HDF locking error")
        else:
            return DatasetOriginal(*args, **kwargs)


def test_handle_hdf_locking_error(dataset_path, monkeypatch, write_proxy):
    """Test that NetCDFWriteProxy can handle non-deterministic HDF locking errors."""
    monkeypatch.setattr(nc, "Dataset", UnreliableDatasetMaker())
    with pytest.raises(OSError, match="Simulated non-deterministic HDF locking error"):
        dataset = nc.Dataset(write_proxy.path, "r+")
        var = dataset.variables[write_proxy.varname]
        var[0] = 1.0

    # Reset.
    monkeypatch.setattr(nc, "Dataset", UnreliableDatasetMaker())
    try:
        write_proxy[0] = 1.0
    except OSError:
        pytest.fail("NetCDFWriteProxy failed to handle HDF locking error")


def test_abandon_many_failures(dataset_path, monkeypatch, write_proxy):
    """Test that NetCDFWriteProxy gives up after many failed attempts."""
    monkeypatch.setattr(
        nc, "Dataset", UnreliableDatasetMaker(attempts_before_success=10)
    )
    with pytest.raises(OSError, match="Simulated non-deterministic HDF locking error"):
        write_proxy[0] = 1.0
