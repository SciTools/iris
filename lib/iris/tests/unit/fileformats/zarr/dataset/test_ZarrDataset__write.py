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


@pytest.fixture
def writable(tmp_path):
    """Return an open, empty, writable dataset."""
    with ZarrDataset(tmp_path / "out.zarr", mode="w-") as dataset:
        yield dataset


class TestCreateDimension:
    def test_a_dimension_is_declared(self, writable):
        writable.create_dimension("time", 3)
        assert dict(writable.dimensions) == {"time": 3}

    def test_declaring_it_twice_at_one_size_is_harmless(self, writable):
        writable.create_dimension("time", 3)
        writable.create_dimension("time", 3)
        assert dict(writable.dimensions) == {"time": 3}

    def test_a_conflicting_size_raises_naming_both(self, writable):
        writable.create_dimension("time", 3)
        with pytest.raises(ValueError, match="time.*3.*4|time.*4.*3"):
            writable.create_dimension("time", 4)

    def test_unlimited_is_refused(self, writable):
        # Spec §4.2: a store with no such concept raises NotImplementedError.
        with pytest.raises(NotImplementedError, match="unlimited"):
            writable.create_dimension("time", None)

    def test_a_read_only_dataset_refuses(self, existing):
        with ZarrDataset(existing) as dataset:
            with pytest.raises(ValueError, match="mode='r'"):
                dataset.create_dimension("time", 3)


class TestCreateVariable:
    def test_an_array_is_created_with_its_dimension_names(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        assert variable.name == "data"
        assert variable.dimensions == ("x",)
        assert variable.shape == (3,)
        assert variable.dtype == np.dtype("f4")

    def test_it_appears_in_the_dataset(self, writable):
        writable.create_dimension("x", 3)
        writable.create_variable("data", np.dtype("f4"), ["x"])
        assert list(writable.variables) == ["data"]

    def test_a_scalar_variable_has_no_dimensions(self, writable):
        variable = writable.create_variable("grid", np.dtype("i4"))
        assert variable.shape == ()

    def test_an_undeclared_dimension_raises_naming_it(self, writable):
        with pytest.raises(ValueError, match="nowhere"):
            writable.create_variable("data", np.dtype("f4"), ["nowhere"])

    def test_an_existing_name_raises_naming_the_group(self, writable):
        # Spec §4.5: "not an overwrite and not a merge".
        writable.create_dimension("x", 3)
        writable.create_variable("data", np.dtype("f4"), ["x"])
        with pytest.raises(ValueError, match="data") as excinfo:
            writable.create_variable("data", np.dtype("f4"), ["x"])
        assert "data" in str(excinfo.value)
        assert writable.location in str(excinfo.value)

    def test_a_read_only_dataset_refuses(self, existing):
        with ZarrDataset(existing) as dataset:
            with pytest.raises(ValueError, match="mode='r'"):
                dataset.create_variable("data", np.dtype("f4"))

    def test_a_string_variable_with_no_fill_value_works(self, writable):
        # dtype=str asks for a variable-length string array, which has no
        # entry in the netCDF default-fill table: there is nothing to look
        # up, so zarr-python's own default is what applies.
        writable.create_dimension("x", 3)
        variable = writable.create_variable("labels", str, ["x"])
        variable[:] = np.array(["a", "bb", "ccc"], dtype=str)
        assert list(variable[:]) == ["a", "bb", "ccc"]


class TestCreateVariableFillValue:
    def test_an_undeclared_fill_is_netcdfs_default(self, writable):
        # Zarr's own default is 0, which would make a masked point of a
        # float array read back as a real zero.
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        assert variable.fill_value == np.float32(9.9692099683868690e36)

    def test_an_undeclared_fill_writes_no_attribute(self, writable):
        # Nothing is missing yet. The saver declares _FillValue when it finds
        # the source is masked.
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        assert "_FillValue" not in variable.attributes

    def test_a_declared_fill_reaches_the_array_and_the_attribute(self, writable):
        import base64
        import struct

        writable.create_dimension("x", 3)
        variable = writable.create_variable(
            "data", np.dtype("f4"), ["x"], fill_value=np.float32(-99.0)
        )
        assert variable.fill_value == np.float32(-99.0)
        packed = base64.b64decode(variable.attributes["_FillValue"])
        assert struct.unpack("<d", packed)[0] == -99.0

    def test_an_integer_fill_is_a_plain_number(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable(
            "data", np.dtype("i2"), ["x"], fill_value=np.int16(-9)
        )
        assert variable.attributes["_FillValue"] == -9


class TestCreateVariableEncoding:
    def test_chunks_and_shards_reach_the_array(self, writable):
        writable.create_dimension("x", 128)
        variable = writable.create_variable(
            "data", np.dtype("f4"), ["x"], chunks=(8,), shards=(64,)
        )
        assert variable.encoding["chunks"] == (8,)
        assert variable.encoding["shards"] == (64,)

    def test_compressors_reach_the_array(self, writable):
        from zarr.codecs import GzipCodec

        writable.create_dimension("x", 8)
        variable = writable.create_variable(
            "data", np.dtype("f4"), ["x"], compressors=[GzipCodec(level=5)]
        )
        codecs = str(variable.encoding["codecs"])
        assert "gzip" in codecs.lower()

    @pytest.mark.parametrize(
        ("keyword", "expected"),
        [
            ("zlib", "compressors"),
            ("complevel", "compressors"),
            ("shuffle", "compressors"),
            ("chunksizes", "chunks"),
            ("fletcher32", "no equivalent"),
            ("endian", "no equivalent"),
            ("unlimited_dimensions", "no equivalent"),
        ],
    )
    def test_a_netcdf_keyword_raises_naming_the_zarr_one(
        self, writable, keyword, expected
    ):
        # Spec §4.5: the saver takes Zarr's vocabulary, and says so rather
        # than translating. The message is the whole user experience here.
        writable.create_dimension("x", 3)
        with pytest.raises(TypeError) as excinfo:
            writable.create_variable("data", np.dtype("f4"), ["x"], **{keyword: True})
        message = str(excinfo.value)
        assert keyword in message
        assert expected in message

    def test_an_unrecognised_keyword_raises_too(self, writable):
        writable.create_dimension("x", 3)
        with pytest.raises(TypeError, match="wibble"):
            writable.create_variable("data", np.dtype("f4"), ["x"], wibble=1)
