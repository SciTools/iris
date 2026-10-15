# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the write half of :class:`iris.fileformats.zarr._dataset.ZarrDataset`."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from iris.fileformats.zarr._dataset import ZarrDataset


@pytest.fixture
def existing(tmp_path):
    """Return the path of a store that already holds one array."""
    path = tmp_path / "existing.zarr"
    group = zarr.open_group(store=path, mode="w-", zarr_format=3)
    array = group.create_array(
        "already_here", shape=(2,), dtype="f4", dimension_names=("x",)
    )
    array[:] = np.array([1.0, 2.0], dtype="f4")
    return path


class TestOpening:
    def test_a_new_store_is_created(self, tmp_path):
        with ZarrDataset(tmp_path / "new.zarr", mode="w-") as dataset:
            assert dataset.mode == "w-"
            assert dataset.format == "ZARR_V3"
        assert (tmp_path / "new.zarr" / "zarr.json").is_file()

    def test_the_default_refuses_an_existing_store(self, existing):
        # The default differs from the netCDF saver's deliberately: a Zarr
        # clobber is many non-atomic deletes, so it is opted into.
        with pytest.raises(FileExistsError, match="mode"):
            ZarrDataset(existing, mode="w-")

    def test_the_refusal_names_the_way_out(self, existing):
        with pytest.raises(FileExistsError, match="'w'"):
            ZarrDataset(existing, mode="w-")

    def test_w_replaces_an_existing_store(self, existing):
        with ZarrDataset(existing, mode="w") as dataset:
            assert list(dataset.variables) == []

    def test_a_adds_to_an_existing_store(self, existing):
        with ZarrDataset(existing, mode="a") as dataset:
            assert list(dataset.variables) == ["already_here"]

    def test_a_opens_a_new_group_in_an_existing_store(self, existing):
        with ZarrDataset(existing, mode="a", group="sub") as dataset:
            assert dataset.location.endswith("existing.zarr/sub")
            assert list(dataset.variables) == []
        # The root is untouched.
        with ZarrDataset(existing) as dataset:
            assert list(dataset.variables) == ["already_here"]

    def test_r_plus_needs_the_store_to_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ZarrDataset(tmp_path / "missing.zarr", mode="r+")

    def test_an_unknown_mode_names_itself(self, tmp_path):
        with pytest.raises(ValueError, match="'a\\+'"):
            ZarrDataset(tmp_path / "new.zarr", mode="a+")

    def test_reading_is_still_the_default(self, existing):
        with ZarrDataset(existing) as dataset:
            assert dataset.mode == "r"


class TestTheVersionGuard:
    @pytest.mark.parametrize("mode", ["w-", "w", "a", "r+"])
    def test_writing_version_2_is_refused(self, tmp_path, mode):
        with pytest.raises(NotImplementedError, match="version 3"):
            ZarrDataset(tmp_path / "new.zarr", mode=mode, zarr_format=2)

    def test_reading_version_2_is_unaffected(self, tmp_path):
        path = tmp_path / "v2.zarr"
        group = zarr.open_group(store=path, mode="w-", zarr_format=2)
        array = group.create_array("data", shape=(2,), dtype="f4")
        array.attrs["_ARRAY_DIMENSIONS"] = ["x"]
        with ZarrDataset(path, mode="r", zarr_format=2) as dataset:
            assert dataset.format == "ZARR_V2"
            assert list(dataset.variables) == ["data"]

    def test_a_version_3_store_is_what_a_write_produces(self, tmp_path):
        with ZarrDataset(tmp_path / "new.zarr", mode="w-") as dataset:
            assert dataset.encoding["zarr_format"] == 3
