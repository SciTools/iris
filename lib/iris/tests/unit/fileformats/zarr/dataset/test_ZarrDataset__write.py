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

    def test_order_is_refused_rather_than_discarded(self, writable):
        # zarr-python takes `order` for version 2 arrays and warns that it
        # has no effect on a version 3 one, which is all Iris writes.
        # Accepting a keyword and then dropping it is worse than refusing it.
        writable.create_dimension("x", 3)
        with pytest.raises(TypeError) as excinfo:
            writable.create_variable("data", np.dtype("f4"), ["x"], order="F")
        message = str(excinfo.value)
        assert "order" in message
        assert "version 2" in message
        assert "version 3" in message

    def test_append_is_told_about_mode(self, writable):
        # There is no append, for Zarr or for netCDF. `mode="a"` adds new
        # arrays, which is the nearest thing and worth naming here.
        writable.create_dimension("x", 3)
        with pytest.raises(TypeError) as excinfo:
            writable.create_variable("data", np.dtype("f4"), ["x"], append=True)
        message = str(excinfo.value)
        assert "append" in message
        assert "mode='a'" in message


class TestAttributeWriting:
    def test_a_numpy_scalar_is_converted(self, writable):
        # zarr's encoder rejects numpy types, and a rejected value stays in
        # the mapping - so converting afterwards is already too late.
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        variable.attributes["scale_factor"] = np.float32(0.5)
        assert variable.attributes["scale_factor"] == 0.5
        assert type(variable.attributes["scale_factor"]) is float

    def test_a_numpy_array_becomes_a_list(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        variable.attributes["valid_range"] = np.array([0.0, 10.0], dtype="f4")
        assert variable.attributes["valid_range"] == [0.0, 10.0]

    def test_bytes_become_text(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        variable.attributes["units"] = b"K"
        assert variable.attributes["units"] == "K"

    def test_a_global_attribute_is_converted_too(self, writable):
        writable.attributes["history"] = np.str_("made by a test")
        assert type(writable.attributes["history"]) is str

    def test_an_unserialisable_attribute_raises_and_writes_nothing(self, writable):
        class Opaque:
            pass

        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        with pytest.raises(TypeError, match="odd"):
            variable.attributes["odd"] = Opaque()
        assert "odd" not in variable.attributes
        # And the store itself is clean, not merely the mapping.
        reopened = zarr.open_group(store=writable.location, mode="r")
        assert "odd" not in dict(reopened["data"].attrs)

    def test_missing_value_is_a_plain_number(self, writable):
        # base64 here opens in xarray and masks nothing.
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        variable.attributes["missing_value"] = np.float32(-99.0)
        assert variable.attributes["missing_value"] == -99.0

    def test_a_fill_value_attribute_is_packed(self, writable):
        import base64
        import struct

        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        variable.attributes["_FillValue"] = np.float32(-99.0)
        packed = base64.b64decode(variable.attributes["_FillValue"])
        assert struct.unpack("<d", packed)[0] == -99.0


class TestValueWriting:
    def test_real_values_are_written(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        variable[:] = np.array([1.0, 2.0, 3.0], dtype="f4")
        np.testing.assert_array_equal(variable[:], [1.0, 2.0, 3.0])

    def test_masked_values_are_filled(self, writable):
        # Without this, zarr dies inside numpy.ma with a message about
        # reshaping, naming neither the cube nor the mask.
        writable.create_dimension("x", 3)
        variable = writable.create_variable(
            "data", np.dtype("f4"), ["x"], fill_value=np.float32(-99.0)
        )
        variable[:] = np.ma.masked_array(
            [1.0, 2.0, 3.0], mask=[False, True, False], dtype="f4"
        )
        np.testing.assert_array_equal(variable[:], [1.0, -99.0, 3.0])

    def test_masked_values_use_the_arrays_own_fill(self, writable):
        writable.create_dimension("x", 2)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable[:] = np.ma.masked_array([1, 2], mask=[False, True], dtype="i2")
        assert variable[:][1] == -32767

    def test_a_string_array_takes_unicode(self, writable):
        writable.create_dimension("n", 2)
        variable = writable.create_variable("labels", str, ["n"])
        variable[:] = np.array(["alpha", "beta"])
        assert list(variable[:]) == ["alpha", "beta"]


class TestValueShapeGuard:
    # zarr-python truncates an oversized assignment without a word, so a
    # caller that sized an array wrongly loses data in silence. The guard
    # turns that into a failure naming the array. See §4.5 of the native
    # Zarr I/O design, `docs/superpowers/specs/2026-09-21-zarr-io-design.md`.
    def test_too_many_values_raises_naming_the_variable(self, writable):
        writable.create_dimension("x", 2)
        variable = writable.create_variable("data", np.dtype("f8"), ["x"])
        with pytest.raises(ValueError, match="data"):
            variable[:] = np.array([1.0, 2.0, 3.0])

    def test_too_many_values_writes_nothing(self, writable):
        writable.create_dimension("x", 2)
        variable = writable.create_variable("data", np.dtype("f8"), ["x"])
        with pytest.raises(ValueError, match="data"):
            variable[:] = np.array([7.0, 8.0, 9.0])
        # Untouched storage, which reads back as the array's own fill.
        np.testing.assert_array_equal(variable[:], [variable.fill_value] * 2)

    def test_the_message_gives_both_shapes(self, writable):
        writable.create_dimension("x", 2)
        variable = writable.create_variable("data", np.dtype("f8"), ["x"])
        with pytest.raises(ValueError, match=r"\(3,\).*\(2,\)"):
            variable[:] = np.array([1.0, 2.0, 3.0])

    def test_too_few_values_raises_too(self, writable):
        # zarr-python raises for this one already, but with a bare numpy
        # broadcasting message naming nothing.
        writable.create_dimension("x", 4)
        variable = writable.create_variable("data", np.dtype("f8"), ["x"])
        with pytest.raises(ValueError, match="data"):
            variable[:] = np.array([1.0, 2.0])

    def test_a_matching_shape_is_written(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f8"), ["x"])
        variable[:] = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(variable[:], [1.0, 2.0, 3.0])

    def test_a_partial_selection_is_measured_against_the_selection(self, writable):
        # What `dask.array.store` does: one block at a time, into a slice.
        writable.create_dimension("x", 4)
        variable = writable.create_variable("data", np.dtype("f8"), ["x"])
        variable[2:4] = np.array([5.0, 6.0])
        np.testing.assert_array_equal(variable[2:4], [5.0, 6.0])

    def test_a_scalar_variable_is_written(self, writable):
        variable = writable.create_variable("data", np.dtype("f8"), [])
        variable[:] = np.array(1.5)
        assert variable[...] == 1.5


class TestPacking:
    # netCDF4-python applies `scale_factor`/`add_offset` packing itself, on
    # assignment, once those attributes are set on the variable - zarr-python
    # does not, so `ZarrDatasetVariable.__setitem__` has to do the arithmetic
    # netCDF gets for free.
    def test_scale_and_offset_pack_the_values(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable.attributes["scale_factor"] = 0.5
        variable.attributes["add_offset"] = 10.0
        variable[:] = np.array([10.0, 10.5, 11.0], dtype="f4")
        np.testing.assert_array_equal(variable[:], [0, 1, 2])

    def test_scale_alone_packs_with_no_offset(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable.attributes["scale_factor"] = 2.0
        variable[:] = np.array([2.0, 4.0, 6.0], dtype="f4")
        np.testing.assert_array_equal(variable[:], [1, 2, 3])

    def test_offset_alone_packs_with_unit_scale(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable.attributes["add_offset"] = 10.0
        variable[:] = np.array([11.0, 12.0, 13.0], dtype="f4")
        np.testing.assert_array_equal(variable[:], [1, 2, 3])

    def test_with_neither_attribute_values_pass_straight_through(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("f4"), ["x"])
        variable[:] = np.array([1.5, 2.5, 3.5], dtype="f4")
        np.testing.assert_array_equal(variable[:], [1.5, 2.5, 3.5])

    def test_packing_rounds_rather_than_truncates(self, writable):
        # `int()` truncates toward zero, which would turn -0.9 into 0 rather
        # than -1. CF packing is round-to-nearest.
        writable.create_dimension("x", 2)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable.attributes["scale_factor"] = 1.0
        variable[:] = np.array([-0.9, 0.9], dtype="f4")
        np.testing.assert_array_equal(variable[:], [-1, 1])

    def test_a_masked_packed_value_keeps_the_arrays_own_fill(self, writable):
        # The sentinel lives in packed (stored) space: the array's own
        # fill_value, not the unpacked value the pack transform would
        # otherwise have produced from it.
        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable.attributes["scale_factor"] = 0.5
        variable.attributes["add_offset"] = 10.0
        variable[:] = np.ma.masked_array(
            [10.0, 10.5, 11.0], mask=[False, True, False], dtype="f4"
        )
        result = variable[:]
        assert result[0] == 0
        assert result[1] == -32767
        assert result[2] == 2

    def test_lazy_packed_values_round_trip(self, writable):
        import dask.array as da

        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable.attributes["scale_factor"] = 0.5
        variable.attributes["add_offset"] = 10.0
        variable[:] = da.from_array(np.array([10.0, 10.5, 11.0], dtype="f4"), chunks=2)
        np.testing.assert_array_equal(variable[:], [0, 1, 2])

    def test_lazy_masked_packed_values_round_trip(self, writable):
        # A real cube's lazy data is a dask array over masked-array blocks -
        # the one combination neither of the two tests above covers alone.
        import dask.array as da

        writable.create_dimension("x", 3)
        variable = writable.create_variable("data", np.dtype("i2"), ["x"])
        variable.attributes["scale_factor"] = 0.5
        variable.attributes["add_offset"] = 10.0
        masked = np.ma.masked_array(
            [10.0, 10.5, 11.0], mask=[False, True, False], dtype="f4"
        )
        variable[:] = da.from_array(masked, chunks=2)
        result = variable[:]
        assert result[0] == 0
        assert result[1] == -32767
        assert result[2] == 2


class TestFinalise:
    def test_it_consolidates(self, writable):
        writable.create_dimension("x", 3)
        writable.create_variable("data", np.dtype("f4"), ["x"])
        writable.finalise()
        reopened = zarr.open_group(store=writable.location, mode="r")
        assert reopened.metadata.consolidated_metadata is not None

    def test_it_warns_about_nothing(self, writable, recwarn):
        # zarr-python warns that consolidated metadata is not in the version 3
        # specification. It is what makes a remote store usable, Iris asks for
        # it deliberately, and the user cannot act on the warning.
        writable.create_dimension("x", 3)
        writable.create_variable("data", np.dtype("f4"), ["x"])
        writable.finalise()
        assert [w for w in recwarn if "onsolidated" in str(w.message)] == []

    def test_a_subgroup_save_consolidates_the_root(self, existing):
        # Spec §4.5: stale consolidated metadata hides the data just written,
        # and a default reader uses it when it is there.
        with ZarrDataset(existing, mode="a", group="sub") as dataset:
            dataset.create_dimension("x", 2)
            dataset.create_variable("data", np.dtype("f4"), ["x"])
            dataset.finalise()
        reopened = zarr.open_group(store=existing, mode="r")
        assert reopened.metadata.consolidated_metadata is not None
        assert "sub" in list(reopened.group_keys())

    def test_a_read_only_dataset_refuses(self, existing):
        with ZarrDataset(existing) as dataset:
            with pytest.raises(ValueError, match="mode='r'"):
                dataset.finalise()
