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
from iris.warnings import IrisCfLoadWarning

from .. import stores


class TestZarrDatasetVersion3(contract.CFDatasetReadContract):
    @pytest.fixture
    def readable(self, tmp_path):
        return ZarrDataset(stores.make_store(tmp_path / "contract.zarr"))


class TestZarrDatasetVersion2(contract.CFDatasetReadContract):
    """The shared contract, plus the one place it sees a version 2 warning.

    The canonical fixture's `air_temperature` sets an explicit storage
    `fill_value` and declares no CF `_FillValue`, so reading it on a version 2
    store warns - see `test_version_2_zeros_are_masked` in
    `loader/test_load_cubes.py` for the rule itself. These two overrides
    exist only to expect that warning; the shared contract has no Zarr
    knowledge to add it generically.
    """

    @pytest.fixture
    def readable(self, tmp_path):
        location = stores.make_store(tmp_path / "contract.zarr", zarr_format=2)
        return ZarrDataset(location)

    def test_read_data_returns_the_data(self, variable):
        with pytest.warns(IrisCfLoadWarning, match="air_temperature"):
            super().test_read_data_returns_the_data(variable)

    def test_read_data_leaves_the_policy_alone_for_a_small_variable(self, variable):
        with pytest.warns(IrisCfLoadWarning, match="air_temperature"):
            super().test_read_data_leaves_the_policy_alone_for_a_small_variable(
                variable
            )
