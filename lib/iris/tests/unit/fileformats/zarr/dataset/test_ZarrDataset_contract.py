# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The shared CFDataset contract, asked of the Zarr implementation.

Both halves. The read half runs against a store built by zarr-python itself,
on each version, so that reading does not depend on Iris having written what
it reads. The write half runs against a store `ZarrDataset` builds through
the CF interface alone - version 3, because that is the only version Iris
writes - and so asks the same questions of the saving path that
`test_NetCDFDataset__contract.py` asks of netCDF's.
"""

import pytest

from iris.fileformats.zarr._dataset import ZarrDataset
from iris.tests.unit.fileformats.cf.dataset import contract

from .. import stores


class TestZarrDatasetVersion3(contract.CFDatasetReadContract):
    chunking_key = "chunks"

    @pytest.fixture
    def readable(self, tmp_path):
        return ZarrDataset(stores.make_store(tmp_path / "contract.zarr"))


class TestZarrDatasetVersion2(contract.CFDatasetReadContract):
    chunking_key = "chunks"

    @pytest.fixture
    def readable(self, tmp_path):
        location = stores.make_store(tmp_path / "contract.zarr", zarr_format=2)
        return ZarrDataset(location)


class TestZarrDatasetWriting(contract.CFDatasetWriteContract):
    """The Zarr side of the write contract."""

    @pytest.fixture
    def writable(self, tmp_path):
        dataset = ZarrDataset(tmp_path / "written.zarr", mode="w-")
        yield dataset
        dataset.close()

    @pytest.fixture
    def reopen(self, tmp_path):
        def _reopen():
            return ZarrDataset(tmp_path / "written.zarr")

        return _reopen
