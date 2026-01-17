# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for string data handling."""

import subprocess

import numpy as np
import pytest

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf import _bytecoding_datasets

# from iris.fileformats.netcdf import _thread_safe_nc
from iris.tests import env_bin_path

NX, N_STRLEN = 3, 64
TEST_STRINGS = ["Münster", "London", "Amsterdam"]
TEST_COORD_VALS = ["bun", "éclair", "sandwich"]

# VARS_COORDS_SHARE_STRING_DIM = True
VARS_COORDS_SHARE_STRING_DIM = False
if VARS_COORDS_SHARE_STRING_DIM:
    # Fix length so that the max coord strlen will be same as data one
    TEST_COORD_VALS[-1] = "Xsandwich"


# Ensure all tests run with "split attrs" turned on.
@pytest.fixture(scope="module", autouse=True)
def enable_split_attrs():
    with iris.FUTURE.context(save_split_attrs=True):
        yield


def convert_strings_to_chararray(string_array_1d, maxlen, encoding="utf-8"):
    bbytes = [text.encode(encoding) for text in string_array_1d]
    pad = b"\0" * maxlen
    bbytes = [(x + pad)[:maxlen] for x in bbytes]
    chararray = np.array([[bb[i : i + 1] for i in range(maxlen)] for bb in bbytes])
    return chararray


def convert_bytesarray_to_strings(
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


INCLUDE_COORD = True
# INCLUDE_COORD = False

INCLUDE_NUMERIC_AUXCOORD = True
# INCLUDE_NUMERIC_AUXCOORD = False


# DATASET_CLASS = _thread_safe_nc.DatasetWrapper
DATASET_CLASS = _bytecoding_datasets.EncodedDataset


def make_testfile(filepath, chararray, coordarray, encoding_str=None):
    ds = DATASET_CLASS(filepath, "w")
    try:
        ds.createDimension("x", NX)
        ds.createDimension("nstr", N_STRLEN)
        vx = ds.createVariable("x", int, dimensions=("x"))
        vx[:] = np.arange(NX)
        if INCLUDE_COORD:
            ds.createDimension("nstr2", N_STRLEN)
            v_co = ds.createVariable(
                "v_co",
                "S1",
                dimensions=(
                    "x",
                    "nstr2",
                ),
            )
            v_co[:] = coordarray
            if encoding_str is not None:
                v_co._Encoding = encoding_str
            if INCLUDE_NUMERIC_AUXCOORD:
                v_num = ds.createVariable(
                    "v_num",
                    float,
                    dimensions=("x",),
                )
                v_num[:] = np.arange(NX)
        v = ds.createVariable(
            "v",
            "S1",
            dimensions=(
                "x",
                "nstr",
            ),
        )
        v[:] = chararray
        if encoding_str is not None:
            v._Encoding = encoding_str
        if INCLUDE_COORD:
            coords_str = "v_co"
            if INCLUDE_NUMERIC_AUXCOORD:
                coords_str += " v_num"
            v.coordinates = coords_str
    finally:
        ds.close()


def make_testcube(
    dataarray,
    coordarray,  # for now, these are always *string* arrays
    encoding_str: str | None = None,
):
    cube = Cube(dataarray, var_name="v")
    cube.add_dim_coord(DimCoord(np.arange(NX), var_name="x"), 0)
    if encoding_str is not None:
        cube.attributes["_Encoding"] = encoding_str
    if INCLUDE_COORD:
        co_x = AuxCoord(coordarray, var_name="v_co")
        if encoding_str is not None:
            co_x.attributes["_Encoding"] = encoding_str
        cube.add_aux_coord(co_x, 0)
    return cube


NCDUMP_PATHSTR = str(env_bin_path("ncdump"))


def ncdump(nc_path: str, *args):
    """Call ncdump to print a dump of a file."""
    call_args = [NCDUMP_PATHSTR, nc_path] + list(args)
    bytes = subprocess.check_output(call_args)
    text = bytes.decode("utf-8")
    print(text)
    return text


def show_result(filepath):
    print(f"File {filepath}")
    print("NCDUMP:")
    ncdump(filepath)
    # with nc.Dataset(filepath, "r") as ds:
    #     v = ds.variables["v"]
    #     print("\n----\nNetcdf data readback (basic)")
    #     try:
    #         print(repr(v[:]))
    #     except UnicodeDecodeError as err:
    #         print(repr(err))
    #     print("..raw:")
    #     v.set_auto_chartostring(False)
    #     print(repr(v[:]))
    print("\nAs iris cube..")
    try:
        iris.loading.LOAD_PROBLEMS.reset()
        cube = iris.load_cube(filepath)
        print(cube)
        if iris.loading.LOAD_PROBLEMS.problems:
            print(iris.loading.LOAD_PROBLEMS)
            print(
                "\n".join(iris.loading.LOAD_PROBLEMS.problems[0].stack_trace.format())
            )
        print("-data-")
        print(repr(cube.data))
        print("-numeric auxcoord data-")
        print(repr(cube.coord("x").points))
        if INCLUDE_COORD:
            print("-string auxcoord data-")
            try:
                print(repr(cube.coord("v_co").points))
            except Exception as err2:
                print(repr(err2))
    except UnicodeDecodeError as err:
        print(repr(err))


@pytest.fixture(scope="session")
def save_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("save_files")


# TODO: the tests don't test things properly yet, they just exercise the code and print
#  things for manual debugging.
tsts = (
    None,
    "ascii",
    "utf-8",
    "utf-32",
)
# tsts = ("utf-8",)
# tsts = ("utf-8", "utf-32",)
# tsts = ("utf-32",)
# tsts = ("utf-8", "ascii", "utf-8")


@pytest.mark.parametrize("encoding", tsts)
def test_load_encodings(encoding, save_dir):
    # small change
    print(f"\n=========\nTesting encoding: {encoding}")
    filepath = save_dir / f"tmp_load_{str(encoding)}.nc"
    do_as = encoding
    if encoding != "utf-32":
        do_as = "utf-8"
    TEST_CHARARRAY = convert_strings_to_chararray(
        TEST_STRINGS, N_STRLEN, encoding=do_as
    )
    TEST_COORDARRAY = convert_strings_to_chararray(
        TEST_COORD_VALS, N_STRLEN, encoding=do_as
    )
    make_testfile(filepath, TEST_CHARARRAY, TEST_COORDARRAY, encoding_str=encoding)
    show_result(filepath)


@pytest.mark.parametrize("encoding", tsts)
def test_save_encodings(encoding, save_dir):
    cube = make_testcube(
        dataarray=TEST_STRINGS, coordarray=TEST_COORD_VALS, encoding_str=encoding
    )
    print(cube)
    filepath = save_dir / f"tmp_save_{str(encoding)}.nc"
    if encoding == "ascii":
        with pytest.raises(
            UnicodeEncodeError,
            match="'ascii' codec can't encode character.*not in range",
        ):
            iris.save(cube, filepath)
    else:
        iris.save(cube, filepath)
        show_result(filepath)
