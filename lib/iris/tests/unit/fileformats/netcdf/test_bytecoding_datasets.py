# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._bytecoding_datasets` module."""

from pathlib import Path

import numpy as np
import pytest

from iris.fileformats.netcdf._bytecoding_datasets import (
    DECODE_TO_STRINGS_ON_READ,
    EncodedDataset,
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


def check_array_matching(arr1, arr2):
    """Check for arrays matching shape, dtype and content."""
    assert (
        arr1.shape == arr2.shape and arr1.dtype == arr2.dtype and np.all(arr1 == arr2)
    )


def check_raw_content(path, varname, expected_byte_array):
    v = fetch_undecoded_var(path, varname)
    bytes_result = v[:]
    check_array_matching(bytes_result, expected_byte_array)


def _make_bytearray_inner(data, bytewidth, encoding):
    # Convert to a (list of [lists of..]) strings or bytes to a
    #  (list of [lists of..]) length-1 bytes with an extra dimension.
    if isinstance(data, str):
        # Convert input strings to bytes
        data = data.encode(encoding)
    if isinstance(data, bytes):
        # iterate over bytes to get a sequence of length-1 bytes (what np.array wants)
        result = [data[i : i + 1] for i in range(len(data))]
        # pad or truncate everything to the required bytewidth
        result = (result + [b"\0"] * bytewidth)[:bytewidth]
    else:
        # If not string/bytes, expect the input to be a list.
        # N.B. the recursion is inefficient, but we don't care about that here
        result = [_make_bytearray_inner(part, bytewidth, encoding) for part in data]
    return result


def make_bytearray(data, bytewidth, encoding="ascii"):
    """Convert bytes or lists of bytes into a numpy byte array.

    This is largely to avoid using "encode_stringarray_as_bytearray", since we don't
    want to depend on that when we should be testing it.
    So, it mostly replicates the function of that, but it does also support bytes in the
    input.
    """
    # First, Convert to a (list of [lists of]..) length-1 bytes objects
    data = _make_bytearray_inner(data, bytewidth, encoding)
    # We should now be able to create an array of single bytes.
    result = np.array(data)
    assert result.dtype == "S1"
    return result


class TestWriteStrings:
    """Test how string data is saved to a file.

    Mostly, we read back data as a "normal" dataset to avoid relying on the read code,
    which is separately tested -- see 'TestReadStrings'.
    """

    def test_encodings(self, encoding, tempdir):
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
        expected_bytes = make_bytearray(writedata, strlen, write_encoding)
        check_raw_content(path, "vxs", expected_bytes)

        # Check also that the "_Encoding" property is as expected
        v = fetch_undecoded_var(path, "vxs")
        result_attr = v.getncattr("_Encoding") if "_Encoding" in v.ncattrs() else None
        assert result_attr == encoding

    def test_scalar(self, tempdir):
        # Like 'test_write_strings', but the variable has *only* the string dimension.
        path = tempdir / "test_writestrings_scalar.nc"

        strlen = 5
        ds_encoded = make_encoded_dataset(path, strlen=strlen)
        v = ds_encoded.createVariable("v0_scalar", "S1", ("strlen",))

        # Checks that we *can* write a string
        v[:] = np.array("stuff", dtype=str)

        # Close, re-open as an "ordinary" dataset, and check the raw content.
        ds_encoded.close()
        expected_bytes = make_bytearray(b"stuff", strlen)
        check_raw_content(path, "v0_scalar", expected_bytes)

    def test_multidim(self, tempdir):
        # Like 'test_write_strings', but the variable has additional dimensions.
        path = tempdir / "test_writestrings_multidim.nc"

        strlen = 5
        ds_encoded = make_encoded_dataset(path, strlen=strlen)
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
        test_data = [
            ["one", "n", ""],
            ["two", "xxxxx", "four"],
        ]
        v[:] = test_data

        # Close, re-open as an "ordinary" dataset, and check the raw content.
        ds_encoded.close()
        expected_bytes = make_bytearray(test_data, strlen)
        check_raw_content(path, "vyxn", expected_bytes)

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

    def test_overlength(self, tempdir):
        # Check expected behaviour with over-length data
        path = tempdir / "test_writestrings_overlength.nc"
        strlen = 5
        ds = make_encoded_dataset(path, strlen=strlen, encoding="ascii")
        v = ds.variables["vxs"]
        v[:] = ["1", "123456789", "two"]
        expected_bytes = make_bytearray(["1", "12345", "two"], strlen)
        check_raw_content(path, "vxs", expected_bytes)

    def test_overlength_splitcoding(self, tempdir):
        # Check expected behaviour when non-ascii multibyte coding gets truncated
        path = tempdir / "test_writestrings_overlength_splitcoding.nc"
        strlen = 5
        ds = make_encoded_dataset(path, strlen=strlen, encoding="utf-8")
        v = ds.variables["vxs"]
        v[:] = ["1", "1234ü", "two"]
        # This creates a problem: it won't read back
        msg = (
            "Character data in variable 'vxs' could not be decoded "
            "with the 'utf-8' encoding."
        )
        with pytest.raises(ValueError, match=msg):
            v[:]

        # Check also that we *can* read the raw content.
        ds.close()
        expected_bytes = [
            b"1",
            b"1234\xc3",  # NOTE: truncated encoding
            b"two",
        ]
        expected_bytearray = make_bytearray(expected_bytes, strlen)
        check_raw_content(path, "vxs", expected_bytearray)


class TestWriteChars:
    @pytest.mark.parametrize("write_form", ["strings", "bytes"])
    def test_write_chars(self, tempdir, write_form):
        encoding = "utf-8"
        write_strings = samples_3_nonascii
        strlen = strings_maxbytes(write_strings, encoding)
        write_bytes = make_bytearray(write_strings, strlen, encoding=encoding)
        # NOTE: 'flexi' form util decides the width needs to be 7 !!
        path = tempdir / f"test_writechars_{write_form}.nc"
        ds = make_encoded_dataset(path, encoding=encoding, strlen=strlen)
        v = ds.variables["vxs"]

        # assign in *either* way..
        if write_form == "strings":
            v[:] = write_strings
        else:
            v[:] = write_bytes

        # .. the result should be the same
        ds.close()
        check_raw_content(path, "vxs", write_bytes)


class TestRead:
    """Test how character data is read and converted to strings.

    N.B. many testcases here parallel the 'TestWriteStrings' : we are creating test
    datafiles with 'make_dataset' and assigning raw bytes, as-per 'TestWriteChars'.

    We are mostly checking here that reading back produces string arrays as expected.
    However, it is simple + convenient to also check the 'DECODE_TO_STRINGS_ON_READ'
    function here, i.e. "raw" bytes reads.  So that is also done in this class.
    """

    @pytest.fixture(params=["strings", "bytes"])
    def readmode(self, request):
        return request.param

    def test_encodings(self, encoding, tempdir, readmode):
        # Create a dataset with the variable
        path = tempdir / f"test_read_encodings_{encoding!s}_{readmode}.nc"

        if encoding in [None, "ascii"]:
            write_strings = samples_3_ascii
            write_encoding = "ascii"
        else:
            write_strings = samples_3_nonascii
            write_encoding = encoding

        write_strings = write_strings.copy()  # just for safety?
        strlen = strings_maxbytes(write_strings, write_encoding)
        write_bytes = make_bytearray(write_strings, strlen, encoding=write_encoding)

        ds_encoded = make_encoded_dataset(path, strlen, encoding)
        v = ds_encoded.variables["vxs"]
        v[:] = write_bytes

        if readmode == "strings":
            # Test "normal" read --> string array
            result = v[:]
            expected = write_strings
            if encoding == "utf-8":
                # In this case, with the given non-ascii sample data, the
                #  "default minimum string length" is overestimated.
                assert strlen == 7 and result.dtype == "U7"
                # correct the result dtype to pass the write_strings comparison below
                truncated_result = result.astype("U4")
                # Also check that content is the same (i.e. not actually truncated)
                assert np.all(truncated_result == result)
                result = truncated_result
        else:
            # Test "raw" read --> byte array
            with DECODE_TO_STRINGS_ON_READ.context(False):
                result = v[:]
            expected = write_bytes

        check_array_matching(result, expected)

    def test_scalar(self, tempdir, readmode):
        # Like 'test_write_strings', but the variable has *only* the string dimension.
        path = tempdir / f"test_read_scalar_{readmode}.nc"

        strlen = 5
        ds_encoded = make_encoded_dataset(path, strlen=strlen)
        v = ds_encoded.createVariable("v0_scalar", "S1", ("strlen",))

        data_string = "stuff"
        data_bytes = make_bytearray(data_string, 5)

        # Checks that we *can* write a string
        v[:] = data_bytes

        if readmode == "strings":
            # Test "normal" read --> string array
            result = v[:]
            expected = np.array(data_string)
        else:
            # Test "raw" read --> byte array
            with DECODE_TO_STRINGS_ON_READ.context(False):
                result = v[:]
            expected = data_bytes

        check_array_matching(result, expected)

    def test_multidim(self, tempdir, readmode):
        # Like 'test_write_strings', but the variable has additional dimensions.
        path = tempdir / f"test_read_multidim_{readmode}.nc"

        strlen = 5
        ds_encoded = make_encoded_dataset(path, strlen=strlen)
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
        test_strings = [
            ["one", "n", ""],
            ["two", "xxxxx", "four"],
        ]
        test_bytes = make_bytearray(test_strings, strlen)
        v[:] = test_bytes

        if readmode == "strings":
            # Test "normal" read --> string array
            result = v[:]
            expected = np.array(test_strings)
        else:
            # Test "raw" read --> byte array
            with DECODE_TO_STRINGS_ON_READ.context(False):
                result = v[:]
            expected = test_bytes

        check_array_matching(result, expected)

    def test_read_encoding_failure(self, tempdir, readmode):
        path = tempdir / f"test_read_encoding_failure_{readmode}.nc"
        strlen = 10
        ds = make_encoded_dataset(path, strlen=strlen, encoding="ascii")
        v = ds.variables["vxs"]
        test_utf8_bytes = make_bytearray(
            samples_3_nonascii, bytewidth=strlen, encoding="utf-8"
        )
        v[:] = test_utf8_bytes

        if readmode == "strings":
            msg = (
                "Character data in variable 'vxs' could not be decoded "
                "with the 'ascii' encoding."
            )
            with pytest.raises(ValueError, match=msg):
                v[:]
        else:
            with DECODE_TO_STRINGS_ON_READ.context(False):
                result = v[:]  # this ought to be ok!

            assert np.all(result == test_utf8_bytes)
