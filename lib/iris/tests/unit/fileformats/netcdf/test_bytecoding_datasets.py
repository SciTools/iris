# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._bytecoding_datasets` module."""

from pathlib import Path

import numpy as np
import pytest

from iris.fileformats.netcdf._bytecoding_datasets import (
    EncodedDataset,
    encode_stringarray_as_bytearray,
    flexi_encode_stringarray_as_bytearray,
)
from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper

encoding_options = [None, "ascii", "utf-8", "utf-32"]

samples_3_ascii = np.array(
    ["one", "", "seven"],  # N.B. include empty!
)
samples_3_nonascii = np.array(["two", "", "épéé"])


def strings_maxbytes(strings, encoding):
    return max(len(string.encode(encoding)) for string in strings)


@pytest.fixture(params=encoding_options)
def encoding(request):
    return request.param


@pytest.fixture(scope="module")
def tempdir(tmp_path_factory):
    path = tmp_path_factory.mktemp("netcdf")
    return path


def make_encoded_dataset(
    path: Path, strlen: int, encoding: str | None = None
) -> EncodedDataset:
    """Create a test EncodedDataset linked to an actual file.

    * strlen becomes the string dimension (i.e. a number of *bytes*)
    * a variable "vxs" is created
    * If 'encoding' is given, the "vxs::_Encoding" attribute is created with this value
    """
    ds = EncodedDataset(path, "w")
    ds.createDimension("x", 3)
    ds.createDimension("strlen", strlen)
    v = ds.createVariable("vxs", "S1", ("x", "strlen"))
    if encoding is not None:
        v.setncattr("_Encoding", encoding)
    return ds


def fetch_undecoded_var(path, varname):
    # Open a path as a "normal" dataset, and return a given variable.
    ds_normal = DatasetWrapper(path)
    ds_normal._contained_instance.set_auto_chartostring(False)
    v = ds_normal.variables[varname]
    # Return a variable, rather than its data, so we can check attributes etc.
    return v


class TestWriteStrings:
    """Test how string data is saved to a file."""

    def test_write_strings(self, encoding, tempdir):
        # Create a dataset with the variable
        path = tempdir / f"test_writestrings_encoding_{encoding!s}.nc"

        if encoding in [None, "ascii"]:
            writedata = samples_3_ascii
            write_encoding = "ascii"
        else:
            writedata = samples_3_nonascii
            write_encoding = encoding

        writedata = writedata.copy()  # just for safety?
        strlen = strings_maxbytes(writedata, write_encoding)

        ds_encoded = make_encoded_dataset(path, strlen, encoding)
        v = ds_encoded.variables["vxs"]

        # Effectively, checks that we *can* write strings
        v[:] = writedata

        # Close, re-open as an "ordinary" dataset, and check the raw content.
        ds_encoded.close()
        v = fetch_undecoded_var(path, "vxs")

        # Check that the raw result is as expected
        bytes_result = v[:]
        expected = encode_stringarray_as_bytearray(writedata, write_encoding, strlen)
        assert (
            bytes_result.shape == expected.shape
            and bytes_result.dtype == expected.dtype
            and np.all(bytes_result == expected)
        )

        # Check that the "_Encoding" property is also as expected
        result_attr = v.getncattr("_Encoding") if "_Encoding" in v.ncattrs() else None
        assert result_attr == encoding

    def test_scalar(self, tempdir):
        # Like 'test_write_strings', but the variable has *only* the string dimension.
        path = tempdir / "test_writestrings_scalar.nc"

        ds_encoded = make_encoded_dataset(path, strlen=5)
        v = ds_encoded.createVariable("v0_scalar", "S1", ("strlen",))

        # Checks that we *can* write a string
        v[:] = np.array("stuff", dtype=str)

        # Close, re-open as an "ordinary" dataset, and check the raw content.
        ds_encoded.close()
        v = fetch_undecoded_var(path, "v0_scalar")
        result = v[:]

        # Check that the raw result is as expected
        assert (
            result.shape == (5,)
            and result.dtype == "<S1"
            and np.all(result == [b"s", b"t", b"u", b"f", b"f"])
        )

    def test_multidim(self, tempdir):
        # Like 'test_write_strings', but the variable has additional dimensions.
        path = tempdir / "test_writestrings_multidim.nc"

        ds_encoded = make_encoded_dataset(path, strlen=5)
        ds_encoded.createDimension("y", 2)
        v = ds_encoded.createVariable(
            "vyxn",
            "S1",
            (
                "y",
                "x",
                "strlen",
            ),
        )

        # Check that we *can* write a multidimensional string array
        test_data = np.array(
            [
                ["one", "n", ""],
                ["two", "xxxxx", "four"],
            ],
            dtype="U5",
        )
        v[:] = test_data

        # Close, re-open as an "ordinary" dataset, and check the raw content.
        ds_encoded.close()
        v = fetch_undecoded_var(path, "vyxn")
        result = v[:]

        # Check that the raw result is as expected
        expected_bytes = encode_stringarray_as_bytearray(
            test_data, encoding="ascii", string_dimension_length=5
        )
        assert (
            result.shape
            == (
                2,
                3,
                5,
            )
            and result.dtype == "<S1"
            and np.all(result == expected_bytes)
        )

    def test_write_encoding_failure(self, tempdir):
        path = tempdir / "test_writestrings_encoding_failure.nc"
        ds = make_encoded_dataset(path, strlen=5, encoding="ascii")
        v = ds.variables["vxs"]
        msg = (
            "String data written to netcdf character variable 'vxs'.*"
            " could not be represented in encoding 'ascii'. "
        )
        with pytest.raises(ValueError, match=msg):
            v[:] = samples_3_nonascii

    def test_overlength_warning(self):
        pass


class TestWriteChars:
    @pytest.mark.parametrize("write_form", ["strings", "bytes"])
    def test_write_chars(self, tempdir, write_form):
        encoding = "utf-8"
        write_strings = samples_3_nonascii
        write_bytes = flexi_encode_stringarray_as_bytearray(
            write_strings, encoding=encoding
        )
        # NOTE: 'flexi' form util decides the width needs to be 7 !!
        strlen = write_bytes.shape[-1]
        path = tempdir / f"test_writechars_{write_form}.nc"
        ds = make_encoded_dataset(path, encoding=encoding, strlen=strlen)
        v = ds.variables["vxs"]

        # assign in *either* way..
        if write_form == "strings":
            v[:] = write_strings
        else:
            v[:] = write_bytes

        # .. the result should be the same
        result = v[:]
        assert (
            result.shape == write_strings.shape
            and result.dtype == f"<U{strlen}"  # NOTE: we fixed the string width
            and np.all(result == write_strings)
        )


class TestReadStrings:
    """Test how character data is read and converted to strings."""

    def test_encodings(self, encoding):
        pass
