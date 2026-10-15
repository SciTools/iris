# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.fileformats.zarr.saver.save`."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.zarr.saver import save


@pytest.fixture
def cubes():
    def one(name):
        cube = Cube(
            np.arange(4, dtype=np.float32).reshape(2, 2),
            standard_name=name,
            units="K",
            var_name=name,
        )
        cube.add_dim_coord(
            DimCoord([0.0, 1.0], standard_name="latitude", units="degrees"), 0
        )
        cube.add_dim_coord(
            DimCoord([0.0, 1.0], standard_name="longitude", units="degrees"), 1
        )
        return cube

    return CubeList([one("air_temperature"), one("air_pressure")])


class TestSave:
    def test_one_cube(self, tmp_path, cubes):
        save(cubes[0], tmp_path / "out.zarr")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert "air_temperature" in list(group.array_keys())

    def test_a_cubelist_goes_into_one_store(self, tmp_path, cubes):
        # One store, opened once: this is why the module joins
        # _CUBELIST_SAVER_MODULES.
        save(cubes, tmp_path / "out.zarr")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert {"air_temperature", "air_pressure"} <= set(group.array_keys())

    def test_the_conventions_attribute_is_written(self, tmp_path, cubes):
        save(cubes[0], tmp_path / "out.zarr")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert group.attrs["Conventions"].startswith("CF-")

    def test_metadata_is_consolidated_by_default(self, tmp_path, cubes):
        save(cubes[0], tmp_path / "out.zarr")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert group.metadata.consolidated_metadata is not None

    def test_consolidation_can_be_declined(self, tmp_path, cubes):
        save(cubes[0], tmp_path / "out.zarr", consolidate_metadata=False)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert group.metadata.consolidated_metadata is None

    def test_mode_a_adds_to_a_consolidated_root(self, tmp_path, cubes):
        # A write mode must see what the store actually holds, not a frozen
        # consolidated snapshot: otherwise a second save believes the store
        # still holds only what the first one wrote.
        save(cubes[0], tmp_path / "out.zarr")
        save(cubes[1], tmp_path / "out.zarr", mode="a")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert {"air_temperature", "air_pressure"} <= set(group.array_keys())


class TestSaveRefusals:
    def test_an_existing_store_is_refused(self, tmp_path, cubes):
        save(cubes[0], tmp_path / "out.zarr")
        with pytest.raises(FileExistsError, match="mode"):
            save(cubes[1], tmp_path / "out.zarr")

    def test_w_replaces(self, tmp_path, cubes):
        save(cubes[0], tmp_path / "out.zarr")
        save(cubes[1], tmp_path / "out.zarr", mode="w")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert "air_temperature" not in list(group.array_keys())

    def test_a_repeated_name_under_a_is_refused(self, tmp_path, cubes):
        # Spec §4.5: not an overwrite, not a merge.
        save(cubes[0], tmp_path / "out.zarr")
        with pytest.raises(ValueError, match="air_temperature"):
            save(cubes[0], tmp_path / "out.zarr", mode="a")

    def test_two_cubes_sharing_a_name_in_one_call_is_not_a_repeat(
        self, tmp_path, cubes
    ):
        # Nothing pre-existed before this call: the first of the pair's own
        # name belongs to this call, not to something the store already
        # held, even under a mode that would refuse a genuine repeat.
        pair = CubeList([cubes[0], cubes[0].copy()])
        save(pair, tmp_path / "out.zarr", mode="a")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert "air_temperature" in list(group.array_keys())
        assert "air_temperature_0" in list(group.array_keys())

    def test_a_netcdf_keyword_is_refused_before_anything_is_created(
        self, tmp_path, cubes
    ):
        # The store must not exist afterwards: a half-made store is worse
        # than no store, and nothing has been written yet.
        with pytest.raises(TypeError, match="zlib"):
            save(cubes[0], tmp_path / "out.zarr", zlib=True)
        assert not (tmp_path / "out.zarr").exists()

    def test_unlimited_dimensions_says_why_not(self, tmp_path, cubes):
        with pytest.raises(TypeError, match="fixed shape"):
            save(cubes[0], tmp_path / "out.zarr", unlimited_dimensions=["latitude"])

    def test_a_non_integer_packing_dtype_is_refused(self, tmp_path, cubes):
        # CF packing's arithmetic only means what it says for an integer
        # target; the netCDF saver refuses the same input with the same
        # message.
        with pytest.raises(ValueError, match="integer"):
            save(cubes[0], tmp_path / "out.zarr", packing="f4")
        assert not (tmp_path / "out.zarr").exists()

    def test_an_incomplete_packing_dict_names_the_missing_key(self, tmp_path, cubes):
        with pytest.raises(ValueError, match="dtype"):
            save(cubes[0], tmp_path / "out.zarr", packing={"scale_factor": 0.1})


class TestSaveIntoAGroup:
    def test_a_group_is_created_and_the_root_is_kept(self, tmp_path, cubes):
        save(cubes[0], tmp_path / "out.zarr")
        save(cubes[1], tmp_path / "out.zarr", group="extra")
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert "air_temperature" in list(group.array_keys())
        assert "air_pressure" in list(group["extra"].array_keys())

    def test_the_root_is_re_consolidated(self, tmp_path, cubes):
        # Spec §4.5: stale consolidated metadata hides what was just written.
        save(cubes[0], tmp_path / "out.zarr")
        save(cubes[1], tmp_path / "out.zarr", group="extra")
        reader = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert "extra" in list(reader.group_keys())
