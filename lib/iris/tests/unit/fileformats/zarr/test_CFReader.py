# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The CF layer reads a Zarr store through the same reader netCDF uses."""

from iris.fileformats.cf import CFReader
from iris.fileformats.zarr._dataset import ZarrDataset

from . import stores


def test_the_cf_reader_classifies_a_zarr_store(tmp_path):
    location = stores.make_store(tmp_path / "s.zarr")
    with CFReader(ZarrDataset(location)) as reader:
        assert "air_temperature" in reader.cf_group.data_variables
        assert "grid" in reader.cf_group.grid_mappings
        assert reader.cf_group.global_attributes["title"] == "contract"
        assert reader.filename == location
