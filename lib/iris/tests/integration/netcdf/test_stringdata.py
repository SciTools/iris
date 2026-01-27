# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for various uses of character/string arrays in netcdf file variables.

This covers both the loading and saving of variables which are the content of
data-variables, auxiliary coordinates, ancillary variables and -possibly?- cell measures.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import iris
from iris.fileformats.netcdf import _thread_safe_nc


@pytest.fixture(scope="module")
def all_lazy_auxcoords():
    """Ensure that *all* aux-coords are loaded lazily, even really small ones."""
    old_minlazybytes = iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES
    iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES = 0
    yield
    iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES = old_minlazybytes


N_XDIM = 3
N_CHARS_DIM = 64
PERSIST_TESTFILES = "~/chararray_testfiles"


NO_ENCODING_STR = "<noencoding>"
TEST_ENCODINGS = [
    NO_ENCODING_STR,
    "ascii",
    "utf-8",
    # "iso8859-1",  # a common one-byte-per-char "codepage" type
    # "utf-16",
    "utf-32",
]


#
# Routines to convert between byte and string arrays.
# Independently defined here, to avoid relying on any code we are testing.
#
def convert_strings_to_chararray(
    string_array_1d: np.ndarray, maxlen: int, encoding: str | None = None
):
    # Note: this is limited to 1-D arrays of strings.
    # Could generalise that if needed, but for now this makes it simpler.
    if encoding is None:
        encoding = "ascii"
    bbytes = [text.encode(encoding) for text in string_array_1d]
    pad = b"\0" * maxlen
    bbytes = [(x + pad)[:maxlen] for x in bbytes]
    chararray = np.array([[bb[i : i + 1] for i in range(maxlen)] for bb in bbytes])
    return chararray


def convert_bytearray_to_strings(
    byte_array, encoding="utf-8", string_length: int | None = None
):
    """Convert bytes to strings.

    N.B. for now at least, we assume the string dim is **always the last one**.
    """
    bytes_shape = byte_array.shape
    var_shape = bytes_shape[:-1]
    if string_length is None:
        string_length = bytes_shape[-1]
    string_dtype = f"U{string_length}"
    result = np.empty(var_shape, dtype=string_dtype)
    for ndindex in np.ndindex(var_shape):
        element_bytes = byte_array[ndindex]
        bytes = b"".join([b if b else b"\0" for b in element_bytes])
        string = bytes.decode(encoding)
        result[ndindex] = string
    return result


@dataclass
class SamplefileDetails:
    """Convenience container for information about a sample file."""

    filepath: Path
    datavar_data: np.ndarray
    stringcoord_data: np.ndarray
    numericcoord_data: np.ndarray


def make_testfile(
    testfile_path: Path,
    encoding_str: str,
    coords_on_separate_dim: bool,
) -> SamplefileDetails:
    """Create a test netcdf file.

    Also returns content information for checking loaded results.
    """
    if encoding_str == NO_ENCODING_STR:
        encoding = None
    else:
        encoding = encoding_str

    data_is_ascii = encoding in (None, "ascii")

    numeric_values = np.arange(3.0)
    if data_is_ascii:
        coordvar_strings = ["mOnster", "London", "Amsterdam"]
        datavar_strings = ["bun", "Eclair", "sandwich"]
    else:
        coordvar_strings = ["Münster", "London", "Amsterdam"]
        datavar_strings = ["bun", "éclair", "sandwich"]

    coordvar_bytearray = convert_strings_to_chararray(
        string_array_1d=coordvar_strings, maxlen=N_CHARS_DIM, encoding=encoding
    )
    datavar_bytearray = convert_strings_to_chararray(
        string_array_1d=datavar_strings, maxlen=N_CHARS_DIM, encoding=encoding
    )

    ds = _thread_safe_nc.DatasetWrapper(testfile_path, "w")
    try:
        ds.createDimension("x", N_XDIM)
        ds.createDimension("nstr", N_CHARS_DIM)
        if coords_on_separate_dim:
            ds.createDimension("nstr2", N_CHARS_DIM)
        v_xdim = ds.createVariable("x", int, dimensions=("x"))
        v_xdim[:] = np.arange(N_XDIM)

        v_co = ds.createVariable(
            "v_co",
            "S1",
            dimensions=(
                "x",
                "nstr2" if coords_on_separate_dim else "nstr",
            ),
        )
        v_co[:] = coordvar_bytearray

        if encoding is not None:
            v_co._Encoding = encoding

        v_numeric = ds.createVariable(
            "v_numeric",
            float,
            dimensions=("x",),
        )
        v_numeric[:] = numeric_values

        v_datavar = ds.createVariable(
            "v",
            "S1",
            dimensions=(
                "x",
                "nstr",
            ),
        )
        v_datavar[:] = datavar_bytearray

        if encoding is not None:
            v_datavar._Encoding = encoding

        v_datavar.coordinates = "v_co v_numeric"
    finally:
        ds.close()

    return SamplefileDetails(
        filepath=testfile_path,
        datavar_data=datavar_strings,
        stringcoord_data=coordvar_strings,
        numericcoord_data=numeric_values,
    )


@pytest.fixture(params=TEST_ENCODINGS)
def encoding(request):
    return request.param


def load_problems_list():
    return [str(prob) for prob in iris.loading.LOAD_PROBLEMS.problems]


class TestReadEncodings:
    """Test loading of testfiles with encoded string data."""

    @pytest.fixture(params=["coordsSameDim", "coordsOwnDim"])
    def use_separate_dims(self, request):
        yield request.param == "coordsOwnDim"

    @pytest.fixture()
    def testdata(
        self,
        encoding,
        tmp_path,
        use_separate_dims,
    ):
        """Create a suitable valid testfile, and return expected string content."""
        if PERSIST_TESTFILES:
            tmp_path = Path(PERSIST_TESTFILES).expanduser()
        if encoding == "<noencoding>":
            filetag = "noencoding"
        else:
            filetag = encoding
        dimtag = "diffdims" if use_separate_dims else "samedims"
        tempfile_path = tmp_path / f"sample_read_{filetag}_{dimtag}.nc"
        testdata = make_testfile(
            testfile_path=tempfile_path,
            encoding_str=encoding,
            coords_on_separate_dim=use_separate_dims,
        )
        from iris.tests.integration.netcdf.test_chararrays import ncdump

        # TODO: temporary for debug -- TO REMOVE
        ncdump(tempfile_path)
        yield testdata

    def test_valid_encodings(self, encoding, testdata: SamplefileDetails):
        testfile_path, datavar_strings, coordvar_strings, numeric_data = (
            testdata.filepath,
            testdata.datavar_data,
            testdata.stringcoord_data,
            testdata.numericcoord_data,
        )
        cube = iris.load_cube(testfile_path)
        assert load_problems_list() == []
        assert cube.shape == (N_XDIM,)

        if encoding != "utf-32":
            expected_string_width = N_CHARS_DIM
        else:
            expected_string_width = (N_CHARS_DIM // 4) - 1
        assert cube.dtype == f"<U{expected_string_width}"
        cube_data = cube.data
        assert np.all(cube_data == datavar_strings)
        coord_var = cube.coord("v_co")
        assert coord_var.dtype == f"<U{expected_string_width}"
        assert np.all(coord_var.points == coordvar_strings)
        # Also check the numeric one.
        coord_var_2 = cube.coord("v_numeric")
        assert coord_var_2.dtype == np.float64
        assert np.all(coord_var_2.points == numeric_data)
