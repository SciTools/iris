# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Run the shared CFDataset contract against the netCDF implementation."""

import pytest

from iris.fileformats.netcdf import _dataset
from iris.tests.unit.fileformats.cf.dataset.contract import (
    CFDatasetReadContract,
    CFDatasetWriteContract,
)


class TestNetCDFDatasetContract(CFDatasetReadContract, CFDatasetWriteContract):
    """The netCDF side of the contract."""

    chunking_key = "chunksizes"

    @pytest.fixture
    def readable(self, tmp_path):
        path = tmp_path / "contract.nc"
        with _dataset.NetCDFDataset(path, mode="w", netcdf_format="NETCDF4") as dataset:
            self.populate(dataset)
        with _dataset.NetCDFDataset(path) as dataset:
            yield dataset

    @pytest.fixture
    def writable(self, tmp_path):
        dataset = _dataset.NetCDFDataset(
            tmp_path / "written.nc", mode="w", netcdf_format="NETCDF4"
        )
        yield dataset
        dataset.close()

    @pytest.fixture
    def reopen(self, tmp_path):
        def _reopen():
            return _dataset.NetCDFDataset(tmp_path / "written.nc")

        return _reopen
