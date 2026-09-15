# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._bytecoding_datasets` module."""

from pathlib import Path
import warnings

import dask.array as da
import netCDF4
import numpy as np
import pytest

from iris import tests
from iris.exceptions import TranslationError
from iris.fileformats.netcdf._bytecoding_datasets import (
    SUPPORTED_ENCODINGS,
    EncodedDataset,
    EncodedGroup,
    EncodedVariable,
    VariableEncoder,
    decode_bytesarray_to_stringarray,
    encode_stringarray_as_bytearray,
)
from iris.fileformats.netcdf._thread_safe_nc import (
    DatasetWrapper,
    GroupWrapper,
    VariableWrapper,
)
import iris.tests._shared_utils as testutils
from iris.tests.integration.netcdf.test_stringdata import (
    convert_bytearray_to_strings,
    convert_strings_to_chararray,
)
from iris.tests.stock.netcdf import ncgen_from_cdl
from iris.warnings import IrisCfLoadWarning, IrisCfSaveWarning

# Note: for test options, include "no encoding" and an alias name
ENCODING_NONE = None
ENCODING_UTF8_ALIAS = "UTF8"
encoding_options = [ENCODING_NONE, ENCODING_UTF8_ALIAS] + SUPPORTED_ENCODINGS

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
    assert arr1.shape == arr2.shape
    assert arr1.dtype == arr2.dtype
    assert np.all(arr1 == arr2)


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
        path = tempdir / f"test_bytecoded_writestrings_encoding_{encoding!s}.nc"

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
        path = tempdir / "test_bytecoded_writestrings_scalar.nc"

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
        path = tempdir / "test_bytecoded_writestrings_multidim.nc"

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
        path = tempdir / f"test_bytecoded_writestrings_encoding_{encoding}_fail.nc"
        ds = make_encoded_dataset(path, strlen=5, encoding="ascii")
        v = ds.variables["vxs"]
        msg = (
            "String data written to netcdf character variable 'vxs'.*"
            f" could not be represented in encoding 'ascii'. "
        )
        with pytest.raises(ValueError, match=msg):
            v[:] = samples_3_nonascii

    @pytest.mark.parametrize("mode", ["invalid", "unsupported"])
    def test_write_badencoding_ignore(self, tempdir, mode):
        if mode == "invalid":
            encoding = "<unknown>"
        else:
            encoding = "latin1"  # "latin1" is a real thing
        path = tempdir / f"test_bytecoded_writestrings_badencoding_{encoding}_ignore.nc"
        ds = make_encoded_dataset(path, strlen=5, encoding=encoding)
        v = ds.variables["vxs"]
        msg = (
            r"Ignoring unsupported encoding for netCDF variable 'vxs': "
            f".*'{encoding}', is not recognised as one of the supported encodings"
        )
        with pytest.warns(IrisCfSaveWarning, match=msg):
            v[:] = samples_3_ascii  # will work OK

    def test_overlength(self, tempdir):
        # Check expected behaviour with over-length data
        path = tempdir / "test_bytecoded_writestrings_overlength.nc"
        strlen = 6
        ds = make_encoded_dataset(path, strlen=strlen, encoding="utf8")
        v = ds.variables["vxs"]
        msg = (
            r"String .* written into netcdf variable 'vxs' with encoding \'utf-8\' "
            r"is 7 bytes long, which exceeds .* 6\. This can be fixed by "
        )
        with pytest.raises(TranslationError, match=msg):
            v[:] = ["1", "éclair", "two"]

    def test_overlength_splitcoding(self, tempdir):
        # Check expected behaviour when non-ascii multibyte coding gets truncated
        path = tempdir / "test_bytecoded_writestrings_overlength_splitcoding.nc"
        strlen = 5
        ds = make_encoded_dataset(path, strlen=strlen, encoding="utf-8")
        v = ds.variables["vxs"]
        # Note: we must do the assignment as a single byte array, to avoid hitting the
        #  safety check for this exact problem : see previous check.
        byte_arrays = [
            string.encode("utf-8")[:strlen] for string in ("1", "1234ü", "two")
        ]
        nd_bytes_array = np.array(
            [
                [bytes[i : i + 1] if i < len(bytes) else b"\0" for i in range(strlen)]
                for bytes in byte_arrays
            ]
        )
        v[:] = nd_bytes_array
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
        path = tempdir / f"test_bytecoded_writechars_{write_form}.nc"
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
    However, each testcase also reads and checks the "raw" byte content by re-opening
    with a non-encoded _thread_safe_nc.DatasetWrapper, to check content is as expected.
    """

    @pytest.fixture(params=["strings", "bytes"])
    def readmode(self, request):
        return request.param

    def undecoded_testvar(self, ds_encoded, varname: str):
        path = ds_encoded.filepath()
        ds_encoded.close()
        ds = DatasetWrapper(path)
        v = ds.variables[varname]
        v.set_auto_chartostring(False)
        return v

    def test_encodings(self, encoding, tempdir, readmode):
        # Create a dataset with the variable
        path = tempdir / f"test_bytecoded_read_encodings_{encoding!s}_{readmode}.nc"

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
            if encoding in ("utf-8", ENCODING_UTF8_ALIAS, "utf-16"):
                # In these cases, with the given non-ascii sample data, the
                #  "default minimum string length" is overestimated.
                if encoding in ["utf-8", ENCODING_UTF8_ALIAS]:
                    assert strlen == 7
                    assert result.dtype == "U7"
                    # correct the result dtype to pass the write_strings comparison below
                    truncated_result = result.astype("U4")
                elif encoding == "utf-16":
                    assert strlen == 10
                    assert result.dtype == "U4"
                    # correct the result dtype to pass the write_strings comparison below
                    truncated_result = result.astype("U4")
                # Also check that content is the same (i.e. not actually truncated)
                assert np.all(truncated_result == result)
                result = truncated_result
        else:
            # Close and re-open as "regular" dataset -- just to check "raw" byte content
            v = self.undecoded_testvar(ds_encoded, "vxs")
            result = v[:]
            expected = write_bytes

        check_array_matching(result, expected)

    def test_scalar(self, tempdir, readmode):
        # Like 'test_write_strings', but the variable has *only* the string dimension.
        path = tempdir / f"test_bytecoded_read_scalar_{readmode}.nc"

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
            v = self.undecoded_testvar(ds_encoded, "v0_scalar")
            result = v[:]
            expected = data_bytes

        check_array_matching(result, expected)

    def test_multidim(self, tempdir, readmode):
        # Like 'test_write_strings', but the variable has additional dimensions.
        path = tempdir / f"test_bytecoded_read_multidim_{readmode}.nc"

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
            v = self.undecoded_testvar(ds_encoded, "vyxn")
            result = v[:]
            expected = test_bytes

        check_array_matching(result, expected)

    def test_read_encoding_failure(self, tempdir, readmode):
        path = tempdir / f"test_bytecoded_read_encoding_failure_{readmode}.nc"
        strlen = 10
        ds_encoded = make_encoded_dataset(path, strlen=strlen, encoding="ascii")
        v = ds_encoded.variables["vxs"]
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
            v = self.undecoded_testvar(ds_encoded, "vxs")
            result = v[:]  # this ought to be ok!

            assert np.all(result == test_utf8_bytes)

    @pytest.mark.parametrize("mode", ["invalid", "unsupported"])
    def test_read_badencoding_ignore(self, tempdir, mode):
        if mode == "invalid":
            encoding = "<unknown>"
        else:
            encoding = "latin1"  # "latin1" is a real thing
        path = tempdir / f"test_bytecoded_read_badencoding_{encoding}_ignore.nc"
        strlen = 10
        ds = make_encoded_dataset(path, strlen=strlen, encoding=encoding)
        v = ds.variables["vxs"]
        test_utf8_bytes = make_bytearray(
            samples_3_nonascii, bytewidth=strlen, encoding="utf-8"
        )
        v[:] = test_utf8_bytes

        msg = (
            r"Ignoring unsupported encoding for netCDF variable 'vxs': "
            f".*'{encoding}', is not recognised as one of the supported encodings"
        )
        with pytest.warns(IrisCfLoadWarning, match=msg):
            # raises warning but succeeds, due to default read encoding of 'utf-8'
            v[:]


class TestObjectTypes:
    """Check that the types of dataset content objects are consistent."""

    @pytest.fixture
    def samplefile_path(self, tmp_path):
        testpath = tmp_path / "test.nc"
        ds = netCDF4.Dataset(testpath, "w")
        ds.createDimension("x", 4)
        grp_a = ds.createGroup("grp_a")
        ds.createVariable("vx", float, ["x"])
        grp_a.createVariable("a_vx", int, ["x"])
        ds.close()
        return testpath

    @pytest.fixture(params=["netCDF4", "unencoded", "encoded"])
    def classtype(self, request):
        param = request.param
        if param == "netCDF4":
            self.dataset_class = netCDF4.Dataset
            self.group_class = netCDF4.Group
            self.variable_class = netCDF4.Variable
        elif param == "unencoded":
            self.dataset_class = DatasetWrapper
            self.group_class = GroupWrapper
            self.variable_class = VariableWrapper
        else:
            self.dataset_class = EncodedDataset
            self.group_class = EncodedGroup
            self.variable_class = EncodedVariable
        return param

    def test_dataset_nonencoded_types(self, samplefile_path, classtype):
        ds = self.dataset_class(samplefile_path)
        try:
            grps = ds.groups
            grp_a = grps["grp_a"]
            assert type(grp_a) is self.group_class
            assert grps == {"grp_a": grp_a}

            var_vx = ds.variables["vx"]
            assert type(var_vx) is self.variable_class

            var_a_vx = grp_a.variables["a_vx"]
            assert type(var_a_vx) is self.variable_class

        finally:
            ds.close()

    @pytest.mark.parametrize("is_on", [True, False], ids=["c2sOn", "c2sOff"])
    @pytest.mark.parametrize("component_type", ["ds", "var", "group"])
    def test_auto_chartostring(self, samplefile_path, classtype, component_type, is_on):
        ds = self.dataset_class(samplefile_path)
        var = ds.variables["vx"]
        grp = ds.groups["grp_a"]
        component = {"ds": ds, "var": var, "group": grp}[component_type]
        if classtype == "encoded" and is_on:
            # In this case cannot turn "on": expect error
            msg = '"auto_chartostring" is not supported by Iris EncodedDataset'
            with pytest.raises(TypeError, match=msg):
                component.set_auto_chartostring(is_on)
        else:
            # Just check method exists +  doesn't error.
            component.set_auto_chartostring(is_on)


# specific tests for underlying support classes
# (not strictly public, but possible for use by ncdata ?
class TestEncodeDecodeFuncs:
    # TODO: make this work with nonstandard chuhks (currently does NOT)

    @pytest.fixture(params=["utf8", "ascii"])
    def encoding(self, request):
        return request.param

    @pytest.fixture(params=["lazy", "concrete"])
    def lazyreal(self, request):
        return request.param

    SAMPLE_STRINGS_UNICODE = [["1", "éclair", "two"]] * 2
    SAMPLE_STRINGS_ASCII = [["1", "bun", "London"]] * 2

    @pytest.fixture(params=["singlechunk", "multichunk"])
    def chunkstyle(self, request):
        return request.param

    def test_encode(self, encoding, lazyreal, chunkstyle):
        if encoding == "ascii":
            strings = self.SAMPLE_STRINGS_ASCII
        else:
            strings = self.SAMPLE_STRINGS_UNICODE

        real_stringarray = np.array(strings, dtype="U10")
        print(
            f"original string data: {real_stringarray!r}, shape={real_stringarray.shape}"
        )
        if lazyreal == "concrete":
            sample_stringarray = real_stringarray
        else:
            chunks = -1 if chunkstyle == "singlechunk" else (1, 3)
            sample_stringarray = da.from_array(
                real_stringarray,
                chunks=chunks,
            )
            print(
                f"sample={sample_stringarray!r}",
                "\n  sample chunksize=",
                sample_stringarray.chunksize,
                "\n  chunks=",
                sample_stringarray.chunks,
            )

        if lazyreal == "concrete":
            result = encode_stringarray_as_bytearray(
                sample_stringarray,
                encoding=encoding,
                string_dimension_length=20,
                var_name="xxx",
            )
        else:
            # For now at least, we need to map the operation ourselves.
            # TODO: do this in encode/decode functions, or VariableEncoder ?
            result = da.map_blocks(
                encode_stringarray_as_bytearray,
                sample_stringarray,
                encoding=encoding,
                string_dimension_length=20,
                var_name="xxx",
                new_axis=2,
                dtype="S1",
            )
            # run the lazy operation
            result = result.compute()

        expected = convert_strings_to_chararray(
            string_array_1d=real_stringarray.reshape((6,)), maxlen=20, encoding=encoding
        ).reshape((2, 3, 20))
        assert np.all(expected == result)

    def test_decode(self, encoding, lazyreal, chunkstyle):
        if encoding == "ascii":
            strings = self.SAMPLE_STRINGS_ASCII
        else:
            strings = self.SAMPLE_STRINGS_UNICODE

        real_stringarray = np.array(strings, dtype="U10")
        real_bytearray = convert_strings_to_chararray(
            string_array_1d=real_stringarray.reshape((6,)), maxlen=20, encoding=encoding
        ).reshape((2, 3, 20))
        print(f"original bytes: {real_bytearray!r}, shape={real_bytearray.shape}")
        if lazyreal == "concrete":
            sample_bytearray = real_bytearray
        else:
            chunks = -1 if chunkstyle == "singlechunk" else (1, 3, -1)
            sample_bytearray = da.from_array(real_bytearray, chunks=chunks)

        if lazyreal == "concrete":
            result = decode_bytesarray_to_stringarray(
                byte_array=sample_bytearray,
                encoding=encoding,
                string_width=20,
                var_name="xxx",
            )
        else:
            # For now at least, we need to map the operation ourselves.
            # TODO: do this in encode/decode functions, or VariableEncoder ?
            result = da.map_blocks(
                decode_bytesarray_to_stringarray,
                sample_bytearray,  # you **can't** make this a named keyword
                encoding=encoding,
                string_width=20,
                var_name="xxx",
                drop_axis=2,
                dtype="U20",  # explicit, as function doesn't support empty arrays
            )
            # run the lazy operation
            result = result.compute()

        expected = real_stringarray.astype("U20")
        assert np.all(expected == result)
