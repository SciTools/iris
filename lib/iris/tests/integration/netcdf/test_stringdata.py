# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for various uses of character/string arrays in netcdf file variables.

This covers both the loading and saving of variables which are the content of
data-variables, auxiliary coordinates, ancillary variables and -possibly?- cell measures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
import pytest

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
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
    string_array_1d: ArrayLike, maxlen: int, encoding: str | None = None
) -> np.ndarray:
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
    byte_array: ArrayLike, encoding: str = "utf-8", string_length: int | None = None
) -> np.ndarray:
    """Convert bytes to strings.

    N.B. for now at least, we assume the string dim is **always the last one**.
    """
    byte_array = np.asanyarray(byte_array)
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
    datavar_data: ArrayLike
    stringcoord_data: ArrayLike
    numericcoord_data: ArrayLike


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
    ) -> Iterable[SamplefileDetails]:
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
        ncdump(str(tempfile_path))
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


@pytest.fixture(params=["stringdata", "bytedata"])
def as_bytes(request):
    yield request.param == "bytedata"


@dataclass
class SampleCubeDetails:
    cube: Cube
    datavar_data: np.ndarray
    stringcoord_data: np.ndarray
    save_path: str | Path | None = None


def make_testcube(
    encoding_str: str | None = None,
    byte_data: bool = False,
) -> SampleCubeDetails:
    data_is_ascii = encoding_str in (NO_ENCODING_STR, "ascii")

    numeric_values = np.arange(3.0)
    if data_is_ascii:
        coordvar_strings = ["mOnster", "London", "Amsterdam"]
        datavar_strings = ["bun", "Eclair", "sandwich"]
    else:
        coordvar_strings = ["Münster", "London", "Amsterdam"]
        datavar_strings = ["bun", "éclair", "sandwich"]

    if not byte_data:
        charlen = N_CHARS_DIM
        if encoding_str == "utf-32":
            charlen = charlen // 4 - 1
        strings_dtype = np.dtype(f"U{charlen}")
        coordvar_array = np.array(coordvar_strings, dtype=strings_dtype)
        datavar_array = np.array(datavar_strings, dtype=strings_dtype)
    else:
        write_encoding = encoding_str
        if write_encoding == NO_ENCODING_STR:
            write_encoding = "ascii"
        coordvar_array = convert_strings_to_chararray(
            coordvar_strings, maxlen=N_CHARS_DIM, encoding=write_encoding
        )
        datavar_array = convert_strings_to_chararray(
            datavar_strings, maxlen=N_CHARS_DIM, encoding=write_encoding
        )

    cube = Cube(datavar_array, var_name="v")
    cube.add_dim_coord(DimCoord(np.arange(N_XDIM), var_name="x"), 0)
    if encoding_str != NO_ENCODING_STR:
        cube.attributes["_Encoding"] = encoding_str
    co_x = AuxCoord(coordvar_array, var_name="v_co")
    if encoding_str != NO_ENCODING_STR:
        co_x.attributes["_Encoding"] = encoding_str
    co_dims = (0, 1) if byte_data else (0,)
    cube.add_aux_coord(co_x, co_dims)

    result = SampleCubeDetails(
        cube=cube,
        datavar_data=datavar_array,
        stringcoord_data=coordvar_array,
    )
    return result


class TestWriteEncodings:
    """Test saving of testfiles with encoded string data.

    To avoid circularity, we generate and save *cube* data.
    """

    @pytest.fixture(params=["dataAsStrings", "dataAsBytes"])
    def write_bytes(self, request):
        yield request.param == "dataAsBytes"

    @pytest.fixture()
    def testpath(self, encoding, write_bytes, tmp_path):
        """Create a suitable test cube, with either string or byte content."""
        if PERSIST_TESTFILES:
            tmp_path = Path(PERSIST_TESTFILES).expanduser()
        if encoding == "<noencoding>":
            filetag = "noencoding"
        else:
            filetag = encoding
        datatag = "writebytes" if write_bytes else "writestrings"
        tempfile_path = tmp_path / f"sample_write_{filetag}_{datatag}.nc"
        yield tempfile_path

    @pytest.fixture()
    def testdata(self, testpath, encoding, write_bytes):
        """Create a suitable test cube + save to a file.

        Apply the given encoding to both coord and cube data.
        Form the data as bytes, or as strings, depending on 'write_bytes'.'
        """
        cube_info = make_testcube(encoding_str=encoding, byte_data=write_bytes)
        cube_info.save_path = testpath
        cube = cube_info.cube
        iris.save(cube, testpath)
        yield cube_info

    def test_valid_encodings(self, encoding, testdata, write_bytes):
        cube_info = testdata
        cube, path = cube_info.cube, cube_info.save_path
        # TODO: not testing the "byte read/write" yet
        # Make a quick check for cube equality : but the presentation depends on the read mode
        # with DECODE_TO_STRINGS_ON_READ.context(not write_bytes):
        # read_cube = iris.load_cube(path)
        # assert read_cube == cube

        # N.B. file content should not depend on whether bytes or strings were written
        vararray, coordarray = cube_info.datavar_data, cube_info.stringcoord_data
        ds = _thread_safe_nc.DatasetWrapper(path)
        ds.set_auto_chartostring(False)
        v_main = ds.variables["v"]
        v_co = ds.variables["v_co"]
        assert v_main.shape == (N_XDIM, N_CHARS_DIM)
        assert v_co.shape == (N_XDIM, N_CHARS_DIM)
        assert v_main.dtype == "<S1"
        assert v_co.dtype == "<S1"
        if encoding == NO_ENCODING_STR:
            assert not "_Encoding" in v_main.ncattrs()
            assert not "_Encoding" in v_co.ncattrs()
        else:
            assert v_main.getncattr("_Encoding") == encoding
            assert v_co.getncattr("_Encoding") == encoding
        data_main = v_main[:]
        data_co = v_co[:]
        if not write_bytes:
            # convert to strings, to compare with originals
            # ("ELSE": varrray/coordarray are bytes anyway)
            if encoding == NO_ENCODING_STR:
                encoding = "ascii"
            data_main = convert_bytearray_to_strings(
                data_main, encoding, string_length=N_CHARS_DIM
            )
            data_co = convert_bytearray_to_strings(
                data_co, encoding, string_length=N_CHARS_DIM
            )
        assert np.all(data_main == vararray)
        assert np.all(data_co == coordarray)
