# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The shared CFDataset read contract, asked of the Zarr implementation.

Only the read half: the Zarr write path does not exist, and `ZarrDataset`
refuses every member of the write surface. The write half of the contract
arrives with the saver.
"""

import pytest

from iris.fileformats.zarr._dataset import ZarrDataset
from iris.tests.unit.fileformats.cf.dataset import contract

from .. import stores


class TestZarrDatasetVersion3(contract.CFDatasetReadContract):
    @pytest.fixture
    def readable(self, tmp_path):
        return ZarrDataset(stores.make_store(tmp_path / "contract.zarr"))


class TestZarrDatasetVersion2(contract.CFDatasetReadContract):
    @pytest.fixture
    def readable(self, tmp_path):
        location = stores.make_store(tmp_path / "contract.zarr", zarr_format=2)
        return ZarrDataset(location)
