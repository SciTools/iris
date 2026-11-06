# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The builders in :mod:`stores` produce the stores the rest of these tests
ask questions of; this checks they build both Zarr versions correctly.
"""

import zarr

from . import stores


def test_make_store_version_3(tmp_path):
    location = stores.make_store(tmp_path / "v3.zarr")
    group = zarr.open_group(store=location, mode="r")
    assert group["air_temperature"].metadata.zarr_format == 3
    assert group["air_temperature"].metadata.dimension_names == ("y", "x")


def test_make_store_version_2(tmp_path):
    location = stores.make_store(tmp_path / "v2.zarr", zarr_format=2)
    group = zarr.open_group(store=location, mode="r")
    array = group["air_temperature"]
    assert array.metadata.zarr_format == 2
    assert array.attrs[stores.ARRAY_DIMENSIONS] == ["y", "x"]
