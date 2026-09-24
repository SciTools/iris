# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the write surface of the netCDF CFDataset implementation."""

import numpy as np
import pytest

from iris.fileformats.netcdf import _bytecoding_datasets, _dataset, _thread_safe_nc


@pytest.fixture
def path(tmp_path):
    return tmp_path / "written.nc"


@pytest.fixture
def writer(path):
    with _dataset.NetCDFDataset(path, mode="w", netcdf_format="NETCDF4") as dataset:
        yield dataset


@pytest.fixture
def grid(writer):
    """A 2x3 grid, ready for variables to be created against."""
    writer.create_dimension("y", 3)
    writer.create_dimension("x", 2)
    return writer


class TestCreateDimension:
    def test_appears_with_its_length(self, writer):
        writer.create_dimension("x", 5)
        assert writer.dimensions["x"] == 5

    def test_none_means_unlimited(self, writer):
        # saver.py:829 passes None for a dimension the user asked to make
        # unlimited. It reads back as zero-length until records are written.
        writer.create_dimension("t", None)
        assert writer.dimensions["t"] == 0
        assert writer.dataset.dimensions["t"].isunlimited()


class TestCreateVariable:
    def test_returns_a_dataset_variable(self, grid):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        assert isinstance(variable, _dataset.NetCDFDatasetVariable)
        assert variable.name == "air"
        assert variable.dimensions == ("y", "x")
        assert variable.shape == (3, 2)
        assert variable.dtype == np.dtype("f4")

    def test_appears_in_the_dataset(self, grid):
        created = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        assert grid.variables["air"].name == created.name

    def test_dimensions_default_to_scalar(self, writer):
        # saver.py:2080 creates a grid-mapping variable with no dimensions at
        # all, and passes only a name and a dtype - see finding F4.
        variable = writer.create_variable("grid", np.int32)
        assert variable.dimensions == ()
        assert variable.shape == ()

    def test_dimensions_may_be_a_list(self, grid):
        # saver.py:1938 and :1983 pass a list, not a tuple.
        variable = grid.create_variable("air", np.dtype("f4"), ["y", "x"])
        assert variable.dimensions == ("y", "x")

    def test_fill_value_becomes_an_attribute(self, grid):
        variable = grid.create_variable(
            "air", np.dtype("f4"), ("y", "x"), fill_value=-1.0
        )
        assert variable.fill_value == -1.0
        assert variable.attributes["_FillValue"] == -1.0

    def test_encoding_is_passed_through(self, grid):
        variable = grid.create_variable(
            "air", np.dtype("f4"), ("y", "x"), zlib=True, chunksizes=(1, 2)
        )
        assert variable.chunking == (1, 2)

    def test_shares_the_datasets_write_lock(self, grid):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        assert variable._write_lock is grid.write_lock


class TestWritingData:
    def test_setitem_round_trip(self, grid, path):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        payload = np.arange(6, dtype="f4").reshape(3, 2)
        variable[:] = payload
        grid.close()

        with _dataset.NetCDFDataset(path) as reader:
            np.testing.assert_array_equal(reader.variables["air"][:], payload)

    def test_write_handle_writes_after_the_dataset_is_closed(self, grid, path):
        # This is what a Dask worker gets as a da.store target, long after the
        # Saver's own handle on the file has gone.
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        handle = variable.write_handle()
        grid.close()

        payload = np.arange(6, dtype="f4").reshape(3, 2)
        handle[:] = payload

        with _dataset.NetCDFDataset(path) as reader:
            np.testing.assert_array_equal(reader.variables["air"][:], payload)

    def test_write_handle_matches_the_variable_encoding(self, grid):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        # The dataset was opened for writing, so its variables are encoded;
        # an unencoded proxy here would silently skip string encoding.
        assert isinstance(
            variable.write_handle(), _bytecoding_datasets.EncodedNetCDFWriteProxy
        )

    def test_write_handle_of_an_unencoded_variable(self, path):
        raw = _thread_safe_nc.DatasetWrapper(path, mode="w", format="NETCDF4")
        try:
            raw.createDimension("x", 2)
            raw.createVariable("air", "f4", ("x",))
            variable = _dataset.NetCDFDatasetVariable(
                raw.variables["air"], str(path), write_lock=None
            )
            assert isinstance(variable.write_handle(), _thread_safe_nc.NetCDFWriteProxy)
        finally:
            raw.close()


class TestSyncAndFinalise:
    def test_sync_flushes(self, grid, path):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        variable[:] = np.zeros((3, 2), dtype="f4")
        grid.sync()
        # Readable while the writer is still open, because sync() flushed.
        with _dataset.NetCDFDataset(path) as reader:
            assert reader.variables["air"].shape == (3, 2)

    def test_finalise_is_a_no_op(self, grid):
        assert grid.finalise() is None
        assert grid.closed is False


class TestAttributeWrites:
    def test_variable_attribute_write_through(self, grid, path):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        variable.attributes["units"] = "K"
        assert variable.attributes["units"] == "K"
        grid.close()

        with _dataset.NetCDFDataset(path) as reader:
            assert reader.variables["air"].attributes["units"] == "K"

    def test_global_attribute_write_through(self, writer, path):
        writer.attributes["Conventions"] = "CF-1.7"
        assert writer.attributes["Conventions"] == "CF-1.7"
        writer.close()

        with _dataset.NetCDFDataset(path) as reader:
            assert reader.attributes["Conventions"] == "CF-1.7"

    def test_attribute_deletion(self, grid, path):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        variable.attributes["units"] = "K"
        del variable.attributes["units"]
        assert "units" not in variable.attributes
        grid.close()

        with _dataset.NetCDFDataset(path) as reader:
            assert "units" not in reader.variables["air"].attributes

    def test_ascii_value_is_written_as_bytes(self, grid):
        # _bytes_if_ascii, moved here from saver.py. netCDF4 writes a bytes
        # value as NC_CHAR; the coercion is what makes every string attribute
        # Iris writes take that type, whatever the file format.
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        variable.attributes["units"] = "K"
        assert variable.variable.getncattr("units") == "K"
        assert _dataset._bytes_if_ascii("K") == b"K"

    def test_non_string_values_pass_through(self, grid):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        variable.attributes["valid_min"] = np.float32(-3.5)
        assert variable.attributes["valid_min"] == np.float32(-3.5)
        assert _dataset._bytes_if_ascii(3) == 3


class TestNonAsciiAttributes:
    """Review Focus 5. The coercion's except branch, which is the common one
    for real-world metadata: degree signs, accented names, superscripts.
    """

    @pytest.mark.parametrize(
        "value",
        [
            "degC \N{DEGREE SIGN}",
            "na\N{LATIN SMALL LETTER I WITH DIAERESIS}ve",
            "m s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}",
        ],
    )
    def test_round_trips_unchanged(self, grid, path, value):
        variable = grid.create_variable("air", np.dtype("f4"), ("y", "x"))
        variable.attributes["long_name"] = value
        assert variable.attributes["long_name"] == value
        grid.close()

        with _dataset.NetCDFDataset(path) as reader:
            assert reader.variables["air"].attributes["long_name"] == value

    def test_is_not_coerced_to_bytes(self):
        value = "degC \N{DEGREE SIGN}"
        assert _dataset._bytes_if_ascii(value) is value

    def test_global_non_ascii_round_trips(self, writer, path):
        value = "Produced at M\N{LATIN SMALL LETTER E WITH ACUTE}t\N{LATIN SMALL LETTER E WITH ACUTE}o"
        writer.attributes["institution"] = value
        writer.close()

        with _dataset.NetCDFDataset(path) as reader:
            assert reader.attributes["institution"] == value
