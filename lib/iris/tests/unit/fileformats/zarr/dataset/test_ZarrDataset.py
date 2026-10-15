# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.zarr._dataset.ZarrDataset`."""

import pytest
import zarr

from iris.fileformats.zarr._dataset import ZarrDataset
from iris.warnings import IrisCfLoadWarning

from .. import stores


@pytest.fixture(params=[2, 3], ids=["v2", "v3"])
def zarr_format(request):
    return request.param


class Test_opening:
    def test_opens_and_reports_itself(self, tmp_path, zarr_format):
        location = stores.make_store(tmp_path / "s.zarr", zarr_format=zarr_format)
        dataset = ZarrDataset(location)

        assert dataset.location == location
        assert dataset.format == f"ZARR_V{zarr_format}"
        assert dataset.mode == "r"
        assert dataset.closed is False
        assert sorted(dataset.variables) == ["air_temperature", "grid"]
        assert dict(dataset.attributes) == stores.GLOBALS
        assert dataset.storage_settings["zarr_format"] == zarr_format


class Test_attributes:
    def test_nc_properties_is_not_a_cf_attribute(self, tmp_path):
        """`_NCProperties` is netCDF-c's own bookkeeping, not CF metadata.

        It is the stamp netCDF-c writes onto the root group of every store it
        creates (confirmed on NCZarr output), reading `version=...,netcdf=...`.
        netCDF4-python hides it from `Dataset.ncattrs()` by convention, so no
        cube loaded through netCDF has ever carried it; a cube loaded from the
        same store through the native Zarr reader must not gain it either, or
        the two would stop comparing equal over nothing more than provenance.
        """
        group = zarr.open_group(store=tmp_path / "nc.zarr", mode="w", zarr_format=3)
        group.attrs["Conventions"] = "CF-1.7"
        group.attrs["_NCProperties"] = "version=2,netcdf=4.10.1,nczarr=2.0.0"
        stores.make_array(group, "v", (2,), "f8", 3, dimensions=("t",))

        dataset = ZarrDataset(str(tmp_path / "nc.zarr"))
        assert "_NCProperties" in group.attrs
        assert "_NCProperties" not in dataset.attributes
        assert dict(dataset.attributes) == {"Conventions": "CF-1.7"}


class Test_dimensions:
    def test_dimensions_are_assembled_from_the_arrays(self, tmp_path, zarr_format):
        location = stores.make_store(tmp_path / "s.zarr", zarr_format=zarr_format)
        assert dict(ZarrDataset(location).dimensions) == stores.DIMENSIONS

    def test_a_name_used_at_two_sizes_is_omitted_and_warns(self, tmp_path):
        """Two arrays may share a dimension name at different lengths.

        The version 3 specification "does not place any restrictions on the use of
        the same dimension name across multiple arrays", so this is conforming
        Zarr and must not raise. Merging the two would produce a wrong cube.

        Opening the store is what reports it. The mapping is assembled in
        `__init__` so that the warning reaches a load, which asks each variable
        for its own `dimensions` and never asks the dataset for this mapping.
        """
        group = zarr.open_group(store=tmp_path / "c.zarr", mode="w", zarr_format=3)
        stores.make_array(group, "short", (2,), "f8", 3, dimensions=("t",))
        stores.make_array(group, "long", (5,), "f8", 3, dimensions=("t",))

        with pytest.warns(IrisCfLoadWarning, match="t"):
            dataset = ZarrDataset(str(tmp_path / "c.zarr"))

        dimensions = dict(dataset.dimensions)
        assert "t" not in dimensions
        # The arrays keep the name in their own dimensions: nothing is renamed or
        # dropped, only the dataset-wide mapping declines to link them.
        assert dataset.variables["short"].dimensions == ("t",)
        assert dataset.variables["long"].dimensions == ("t",)

    def test_anonymous_dimensions_are_not_in_the_mapping(self, tmp_path):
        group = zarr.open_group(store=tmp_path / "a.zarr", mode="w", zarr_format=3)
        stores.make_array(group, "named", (2,), "f8", 3, dimensions=("t",))
        stores.make_array(group, "unnamed", (4,), "f8", 3)

        dataset = ZarrDataset(str(tmp_path / "a.zarr"))
        assert dict(dataset.dimensions) == {"t": 2}


class Test_close:
    def test_close_is_idempotent(self, tmp_path):
        location = stores.make_store(tmp_path / "s.zarr")
        dataset = ZarrDataset(location)
        dataset.close()
        dataset.close()
        assert dataset.closed is True

    def test_context_manager_closes(self, tmp_path):
        location = stores.make_store(tmp_path / "s.zarr")
        dataset = ZarrDataset(location)
        with dataset as entered:
            assert entered is dataset
        assert dataset.closed is True


class Test_groups:
    def test_group_keyword_selects_a_subgroup(self, tmp_path):
        root = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
        sub = root.create_group("measurements/r10m")
        stores.make_array(sub, "b02", (2,), "u2", 3, dimensions=("p",))

        dataset = ZarrDataset(str(tmp_path / "n.zarr"), group="measurements/r10m")
        assert list(dataset.variables) == ["b02"]
        assert dataset.location.endswith("measurements/r10m")

    def test_subgroup_names_lists_only_the_groups(self, tmp_path):
        root = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
        root.create_group("conditions")
        root.create_group("measurements")
        stores.make_array(root, "v", (2,), "f8", 3)

        dataset = ZarrDataset(str(tmp_path / "n.zarr"))
        assert sorted(dataset.subgroup_names) == ["conditions", "measurements"]

    def test_nested_groups_are_not_walked(self, tmp_path):
        """Iris loads one group per call, like xarray.

        A Zarr store can legitimately hold unrelated datasets, and merging them
        into one CubeList would be wrong.
        """
        root = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
        stores.make_array(root, "top", (2,), "f8", 3)
        stores.make_array(root.create_group("sub"), "inner", (2,), "f8", 3)

        assert list(ZarrDataset(str(tmp_path / "n.zarr")).variables) == ["top"]

    def test_a_missing_group_raises(self, tmp_path):
        location = stores.make_store(tmp_path / "s.zarr")
        with pytest.raises(zarr.errors.GroupNotFoundError):
            ZarrDataset(location, group="nope")


class Test_supports_dtype:
    @pytest.mark.parametrize(
        "dtype", ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "c8"]
    )
    def test_supports_every_dtype(self, tmp_path, zarr_format, dtype):
        # The netCDF 3 formats have no 64-bit or unsigned integer type, which is
        # what makes supports_dtype worth asking. Zarr has every one of them, in
        # both versions, so the Zarr answer carries no format test.
        dataset = ZarrDataset(
            stores.make_store(tmp_path / "s.zarr", zarr_format=zarr_format)
        )
        assert dataset.supports_dtype(np.dtype(dtype)) is True


class Test_locations:
    def test_an_in_memory_store_is_just_another_location(self, tmp_path):
        """A store need not be a directory. zarr-python decides what it is."""
        memory = zarr.storage.MemoryStore()
        group = zarr.open_group(store=memory, mode="w", zarr_format=3)
        stores.make_array(group, "v", (2,), "f8", 3, dimensions=("t",))

        dataset = ZarrDataset(memory)
        assert list(dataset.variables) == ["v"]

    def test_a_missing_store_raises_zarrs_own_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ZarrDataset(str(tmp_path / "nothing.zarr"))

    def test_a_directory_that_is_not_a_store_raises(self, tmp_path):
        (tmp_path / "plain").mkdir()
        with pytest.raises(zarr.errors.GroupNotFoundError):
            ZarrDataset(str(tmp_path / "plain"))
