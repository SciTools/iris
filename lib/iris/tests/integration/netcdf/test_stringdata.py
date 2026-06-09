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
import iris.exceptions
from iris.fileformats.netcdf import (
    DECODE_TO_STRINGS_ON_READ,
    SUPPORTED_ENCODINGS,
    _thread_safe_nc,
)


@pytest.fixture(scope="module")
def all_lazy_auxcoords():
    """Ensure that *all* aux-coords are loaded lazily, even really small ones."""
    old_minlazybytes = iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES
    iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES = 0
    yield
    iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES = old_minlazybytes


N_XDIM = 3
N_CHARS_DIM = 64

NO_ENCODING_STR = "<noencoding>"
ALIAS_UTF8_STR = "UTF8"  # an alternative acceptable form (should be written as-is)
TEST_ENCODINGS = [NO_ENCODING_STR, ALIAS_UTF8_STR] + SUPPORTED_ENCODINGS


# Common fixture to save with split-attrs ONLY in these tests
@pytest.fixture(scope="module", autouse=True)
def all_split_attrs():
    with iris.FUTURE.context(save_split_attrs=True):
        yield


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
        bytes = b"".join([b or b"\0" for b in element_bytes])
        string = bytes.decode(encoding)
        result[ndindex] = string
    return result


@dataclass
class SamplefileDetails:
    """Convenience container for information about a sample file."""

    filepath: Path
    datavar_data: ArrayLike
    datavar_bytes: ArrayLike
    stringcoord_data: ArrayLike
    stringcoord_bytes: ArrayLike
    numericcoord_data: ArrayLike


def make_testfile(
    testfile_path: Path,
    encoding_str: str,
    coords_on_separate_dim: bool = False,
    # If set, determines the "_Encoding" attrs content, including None --> no attr.
    # Otherwise, they  follow 'encoding_str', including NO_ENCODING_STR --> no attr.
    encoding_attr: str | None = "<as_encoding_str>",
) -> SamplefileDetails:
    """Create a test netcdf file.

    Also returns content information for checking loaded results.
    """
    if encoding_str == NO_ENCODING_STR:
        encoding = None
    else:
        encoding = encoding_str

    if encoding_attr == "<as_encoding_str>":
        encoding_attr = encoding

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

        if encoding_attr is not None:
            v_co._Encoding = encoding_attr

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

        if encoding_attr is not None:
            v_datavar._Encoding = encoding_attr

        v_datavar.coordinates = "v_co v_numeric"
    finally:
        ds.close()

    return SamplefileDetails(
        filepath=testfile_path,
        datavar_data=datavar_strings,
        datavar_bytes=datavar_bytearray,
        stringcoord_data=coordvar_strings,
        stringcoord_bytes=coordvar_bytearray,
        numericcoord_data=numeric_values,
    )


@pytest.fixture(params=TEST_ENCODINGS)
def encoding(request):
    return request.param


def load_problems_list():
    return [str(prob) for prob in iris.loading.LOAD_PROBLEMS.problems]


class TestReadEncodings:
    """Test loading of testfiles with encoded string data."""

    @pytest.fixture(autouse=True)
    def _clear_load_problems(self):
        iris.loading.LOAD_PROBLEMS.reset()
        return

    @pytest.fixture(params=["coordsSameDim", "coordsOwnDim"])
    def use_separate_dims(self, request):
        return request.param == "coordsOwnDim"

    @pytest.fixture
    def readtest_path(
        self,
        encoding,
        tmp_path,
        use_separate_dims,
    ) -> Iterable[SamplefileDetails]:
        """Create a suitable valid testfile, and return expected string content."""
        if encoding == "<noencoding>":
            filetag = "noencoding"
        else:
            filetag = encoding
        dimtag = "diffdims" if use_separate_dims else "samedims"
        tempfile_path = tmp_path / f"sample_stringdata_read_{filetag}_{dimtag}.nc"
        return tempfile_path

    @pytest.fixture
    def readtest_data(
        self,
        encoding,
        readtest_path,
        use_separate_dims,
    ) -> SamplefileDetails:
        """Create a suitable valid testfile, and return expected string content."""
        testdata = make_testfile(
            testfile_path=readtest_path,
            encoding_str=encoding,
            coords_on_separate_dim=use_separate_dims,
        )
        return testdata

    @pytest.fixture(params=["strings", "bytes"])
    def readmode(self, request):
        return request.param

    def test_valid_encodings(
        self, encoding, readtest_data: SamplefileDetails, readmode, use_separate_dims
    ):
        (
            testfile_path,
            datavar_strings,
            datavar_bytes,
            coordvar_strings,
            coordvar_bytes,
            numeric_data,
        ) = (
            readtest_data.filepath,
            readtest_data.datavar_data,
            readtest_data.datavar_bytes,
            readtest_data.stringcoord_data,
            readtest_data.stringcoord_bytes,
            readtest_data.numericcoord_data,
        )

        if readmode == "bytes" and use_separate_dims == True:
            msg = (
                "Unsupported load combination : character coordinates with a non-cube "
                "string dimension can't attach to the cube, when read as bytes."
            )
            pytest.skip(msg)

        as_strings = readmode == "strings"
        if as_strings:
            # Regular load
            cube = iris.load_cube(testfile_path)
            expected_shape: tuple = (N_XDIM,)
        else:
            # Special NON-decoded read
            with DECODE_TO_STRINGS_ON_READ.context(False):
                cube = iris.load_cube(testfile_path)
            expected_shape = (N_XDIM, N_CHARS_DIM)

        assert load_problems_list() == []
        assert cube.shape == expected_shape

        if as_strings:
            if encoding == "utf-32":
                expected_string_width = (N_CHARS_DIM // 4) - 1
            elif encoding == "utf-16":
                expected_string_width = (N_CHARS_DIM) // 2 - 1
            else:
                expected_string_width = N_CHARS_DIM
            expected_dtype = f"<U{expected_string_width}"
        else:
            expected_dtype = "S1"
        assert cube.dtype == expected_dtype

        cube_data = cube.data
        expected_data = datavar_strings if as_strings else datavar_bytes
        assert np.all(cube_data == expected_data)

        coord_var = cube.coord("v_co")
        assert coord_var.dtype == expected_dtype
        expected_points = coordvar_strings if as_strings else coordvar_bytes
        assert np.all(coord_var.points == expected_points)

        # Also check the numeric one.
        coord_var_2 = cube.coord("v_numeric")
        assert coord_var_2.dtype == np.float64
        assert np.all(coord_var_2.points == numeric_data)


@pytest.fixture(params=["stringdata", "bytedata"])
def as_bytes(request):
    return request.param == "bytedata"


@dataclass
class SampleCubeDetails:
    cube: Cube
    datavar_data: np.ndarray
    stringcoord_data: np.ndarray
    save_path: str | Path | None = None


def make_testcube(
    encoding_str: str | None = None,
    byte_data: bool = False,
    lazy_data: bool = False,
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
        # Do our own conversion between intended byte dimension and string width
        # N.B. N_CHARS_DIM is set big enough so the test strings will never overflow
        charlen = N_CHARS_DIM
        if encoding_str == "utf-32":
            charlen = (charlen // 4) - 1
        elif encoding_str == "utf-16":
            charlen = (charlen // 2) - 1
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

    if lazy_data:
        from iris._lazy_data import as_lazy_data

        datavar_array, coordvar_array = (
            as_lazy_data(arr) for arr in [datavar_array, coordvar_array]
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

    @pytest.fixture(params=["allLazy", "smallReal"])
    def lazy_data(self, request, mocker):
        is_lazy = request.param == "allLazy"
        if is_lazy:
            mocker.patch("iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES", 0)
        return is_lazy

    @pytest.fixture(params=["dataAsStrings", "dataAsBytes"])
    def write_bytes(self, request):
        return request.param == "dataAsBytes"

    @pytest.fixture
    def writetest_path(self, encoding, write_bytes, lazy_data, tmp_path):
        """Create a suitable test cube, with either string or byte content."""
        if encoding == "<noencoding>":
            filetag = "noencoding"
        else:
            filetag = encoding
        datatag = "writebytes" if write_bytes else "writestrings"
        lazytag = "alllazy" if lazy_data else "smallreal"
        tempfile_path = (
            tmp_path / f"sample_stringdata_write_{filetag}_{datatag}_{lazytag}.nc"
        )
        return tempfile_path

    @pytest.fixture
    def writetest_data(self, writetest_path, encoding, write_bytes, lazy_data):
        """Create a suitable test cube + save to a file.

        Apply the given encoding to both coord and cube data.
        Form the data as bytes, or as strings, depending on 'write_bytes'.'
        """
        cube_info = make_testcube(
            encoding_str=encoding,
            byte_data=write_bytes,
            lazy_data=lazy_data,
        )
        cube_info.save_path = writetest_path
        cube = cube_info.cube
        iris.save(cube, writetest_path)
        return cube_info

    def test_valid_encodings(self, encoding, writetest_data, write_bytes):
        cube_info = writetest_data
        cube, path = cube_info.cube, cube_info.save_path

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


class TestStringCubeBehaviour:
    def test_create(self):
        cube = Cube(["this", "that", "cliché"])
        assert isinstance(cube.core_data(), np.ndarray)
        assert cube.shape == (3,)
        assert cube.dtype == np.dtype("U6")

    def test_scalar_extract(self):
        cube = Cube(["one", "two", "thirteen"])
        cube = cube[0]
        assert isinstance(cube.core_data(), np.ndarray)
        assert cube.shape == ()
        assert cube.dtype == np.dtype("U3")

    def test_scalar_create(self):
        cube = Cube("éclair")
        assert isinstance(cube.core_data(), np.ndarray)
        assert cube.shape == ()
        assert cube.dtype == np.dtype("U6")


class TestWriteReadMixedEncodings:
    """Check saving of different types of string data, in cubes.

    Checks that encodings are preserved through save/load.
    Checks that scalar cubes save.
    Checks that multiple cubes with different encodings save correctly.
    """

    def test_mixed(self, tmp_path):
        # Save a mixture of string + numeric cubes, 1-D and scalar
        # Ensure that they save, and read back correctly.
        c1 = Cube(["test-string"], var_name="c1")
        c2 = Cube(["test=éclair"], var_name="c2", attributes={"_Encoding": "utf16"})
        c3 = Cube(4.5, var_name="c3")
        c4 = Cube(np.array("q"), var_name="c4")  # a SCALAR character-type cube
        cubes = [c1, c2, c3, c4]
        originals = [c.copy() for c in cubes]

        # Check they save OK
        filepath = tmp_path / "tst.nc"
        iris.save(cubes, filepath)

        # Check they also read back the same (except for Conventions attribute)
        results = iris.load_cubes(filepath, ["c1", "c2", "c3", "c4"])
        for cube in results:
            cube.attributes.pop("Conventions", None)
        assert all(orig == result for orig, result in zip(originals, results))


class TestWriteReadScalarStringCubes:
    """Check how scalar string-typed cubes are saved.
    NB all these gain a string dimension, even when only a single byte character,
    so they are not actually "scalar" in the file.
    """

    def test_save_scalar_ascii__ok(self, tmp_path):
        # We can save a scalar cube containing a *single ascii character*
        scalar_char_cube = Cube(
            np.array("x"),
            var_name="c1",
            attributes={"_Encoding": "utf8"},  # NB no encoding is *needed* here.
        )
        assert scalar_char_cube.shape == ()
        filepath = tmp_path / "tst.nc"
        iris.save(scalar_char_cube, filepath)

        # Check dims in file
        ds = _thread_safe_nc.DatasetWrapper(filepath)
        assert ds.variables["c1"].dimensions == ("string1",)
        assert ds.dimensions["string1"].size == 1
        ds.close()

        # check read-back result
        result = iris.load_cube(filepath)
        result.attributes.pop("Conventions", None)
        assert result == scalar_char_cube

    def test_save_scalar_unicode__fail(self, tmp_path):
        # You *can't* save a scalar cube containing a non-ascii character
        # *without an explicitly lengthened dtype*,
        # because it doesn't convert to a single "char".
        scalar_char_bad = Cube(
            np.array("ü"), var_name="c1", attributes={"_Encoding": "utf8"}
        )
        assert scalar_char_bad.shape == ()
        filepath = tmp_path / "tst.nc"
        msg = (
            "String 'ü' written .* is 2 bytes long, "
            "which exceeds the string dimension length"
        )
        with pytest.raises(iris.exceptions.TranslationError, match=msg):
            iris.save(scalar_char_bad, filepath)

    def test_save_single_unicode__okay(self, tmp_path):
        # You *can* save a scalar cube containing a non-ascii character,
        # *if* the dtype is extended to allow for multiple encoded bytes.
        scalar_char_cube = Cube(
            np.array("ü", dtype="U2"), var_name="c1", attributes={"_Encoding": "utf8"}
        )
        assert scalar_char_cube.shape == ()
        filepath = tmp_path / "tst.nc"
        iris.save(scalar_char_cube, filepath)

        # Check dims in file
        ds = _thread_safe_nc.DatasetWrapper(filepath)
        assert ds.variables["c1"].dimensions == ("string2",)
        assert ds.dimensions["string2"].size == 2
        ds.close()

        # check read-back result
        result = iris.load_cube(filepath)
        result.attributes.pop("Conventions", None)
        assert result == scalar_char_cube


class TestReadParticularCases:
    @pytest.mark.parametrize("data_encoding", ["utf8", "utf16", "utf32"])
    def test_read_no_encoding(self, tmp_path, data_encoding):
        # Check that we can read UTF-8 encoded data, even with no _Encoding attribute.
        # This is a common case in the wild, and now accepted by CF as a default.
        # However, other encodings will FAIL to decode.
        filepath = tmp_path / "utf8_no_encoding.nc"
        testdata = make_testfile(
            testfile_path=filepath,
            encoding_str=data_encoding,
            encoding_attr=None,
        )
        cube = iris.load_cube(filepath)
        assert "_Encoding" not in cube.attributes

        if data_encoding == "utf8":
            assert np.all(cube.data == testdata.datavar_data)
        else:
            # NOTE: no error on loading, only when you fetch content + it decodes.
            msg = "Character data .* could not be decoded with the 'utf-8' encoding"
            with pytest.raises(ValueError, match=msg):
                cube.data

    def test_read_wrong_encoding__fail(self, tmp_path):
        filepath = tmp_path / "missing_encoding.nc"
        testdata = make_testfile(
            testfile_path=filepath,
            encoding_str="utf-16",
            encoding_attr="utf-8",
        )
        cube = iris.load_cube(filepath)
        # NOTE: no error on loading, only when you fetch content + it decodes.
        msg = "Character data .* could not be decoded with the 'utf-8' encoding."
        with pytest.raises(ValueError, match=msg):
            data = cube.data


class TestWriteParticularCases:
    def test_write_unicode_no_encoding__fail(self, tmp_path):
        cube = Cube(np.array("éclair"))
        filepath = tmp_path / "write_unicode_no_encoding.nc"
        msg = (
            "String data written to netcdf character variable 'unknown' "
            "could not be represented in encoding 'ascii'"
        )
        with pytest.raises(ValueError, match=msg):
            iris.save(cube, filepath)

    def test_write_encoded_overlength__fail(self, tmp_path):
        cube = Cube(np.array("éclair"), attributes={"_Encoding": "utf8"})
        filepath = tmp_path / "write_encoded_overlength.nc"
        msg = (
            "String 'éclair' written into netcdf variable 'unknown' "
            "with encoding 'utf-8' is 7 bytes long, which exceeds the "
            "string dimension length, 6. "
            r"This can be fixed by converting the data to a \"wider\" string dtype, "
            r"e.g. cube.data = cube.data.astype\(\"U7\"\)"
        )
        with pytest.raises(iris.exceptions.TranslationError, match=msg):
            iris.save(cube, filepath)

    def test_write_multibytes__fail(self, tmp_path):
        encoded_bytes = "éclair".encode("utf8")
        byte_array = np.array(encoded_bytes)
        cube = Cube(byte_array, attributes={"_Encoding": "utf8"})
        filepath = tmp_path / "write_multibyte_Sxx.nc"
        msg = (
            r"Variable 'unknown' has unexpected dtype, dtype\('S7'\)."
            "Data content arrays must be numeric, or contain single-bytes "
            r"\(dtype 'S1'\), or unicode strings \(dtype 'U<n>'\)."
        )
        with pytest.raises(ValueError, match=msg):
            iris.save(cube, filepath)

    def test_write_stringobjects__fail(self, tmp_path):
        string_array = np.array(["one", "four"], dtype="O")
        cube = Cube(string_array)
        filepath = tmp_path / "write_stringobjects.nc"
        msg = (
            r"Variable 'unknown' has unexpected dtype, dtype\('O'\)."
            "Data content arrays must be numeric, or contain single-bytes "
            r"\(dtype 'S1'\), or unicode strings \(dtype 'U<n>'\)."
        )
        with pytest.raises(ValueError, match=msg):
            iris.save(cube, filepath)

    def test_write_unexpected_dtype_itemsize(self, mocker, tmp_path):
        # Test unexpected form of numpy character data.  Not clear if this can actually
        #  happen, but we do have a runtime test for it, so this just exercises that.
        mock_dtype = mocker.Mock(spec=np.dtype, kind="U", itemsize=3)
        mock_data = mocker.MagicMock(spec=np.ndarray, dtype=mock_dtype)
        mocker.patch("numpy.asarray", return_value=mock_data)
        cube = Cube(mock_data)
        filepath = tmp_path / "write_unexpected_dtype_itemsize.nc"
        msg = (
            r"Unexpected numpy string 'dtype\.itemsize' for element 'unknown': "
            r"'dtype\.itemsize = 3, expected a multiple of four \(always\)\."
        )
        with pytest.raises(ValueError, match=msg):
            iris.save(cube, filepath)


class TestSaveloadBadUnicodeAsBytes:
    def test_save_load_bad_unicode(self, tmp_path):
        filepath = tmp_path / "bad_unicode_utf8.nc"
        test_string = "marré"
        bytes_array = test_string.encode("utf8")
        s1_array = np.array([bytes_array[i : i + 1] for i in range(len(bytes_array))])
        s1_array_bad_utf8 = s1_array[:-1]  # invalid without the last byte
        cube = Cube(s1_array_bad_utf8, attributes={"_Encoding": "utf8"})
        iris.save(cube, filepath)
        # First check for error when reading back *normally*
        msg = "could not be decoded with the 'utf-8' encoding"
        with pytest.raises(ValueError, match=msg):
            iris.load(filepath)
        # .. but OK in byte-reading mode
        with iris.fileformats.netcdf.DECODE_TO_STRINGS_ON_READ.context(False):
            readback_cube = iris.load_cube(filepath)
        assert readback_cube.dtype == "S1"
        assert np.all(readback_cube.data == s1_array_bad_utf8)
