import netCDF4 as nc
import numpy as np
import pytest

import iris

iris.FUTURE.save_split_attrs = True
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

NX, N_STRLEN = 3, 64
TEST_STRINGS = ["Münster", "London", "Amsterdam"]
TEST_COORD_VALS = ["bun", "éclair", "sandwich"]

# VARS_COORDS_SHARE_STRING_DIM = True
VARS_COORDS_SHARE_STRING_DIM = False
if VARS_COORDS_SHARE_STRING_DIM:
    TEST_COORD_VALS[-1] = "Xsandwich"  # makes the max coord strlen same as data one


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


def make_testfile(filepath, chararray, coordarray, encoding_str=None):
    with nc.Dataset(filepath, "w") as ds:
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


def show_result(filepath):
    from pp_utils import ncdump

    print(f"File {filepath}")
    print("NCDUMP:")
    ncdump(filepath, "")
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
def test_load_encodings(encoding):
    # small change
    print(f"\n=========\nTesting encoding: {encoding}")
    filepath = f"tmp_{str(encoding)}.nc"
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
def test_save_encodings(encoding):
    cube = make_testcube(
        dataarray=TEST_STRINGS, coordarray=TEST_COORD_VALS, encoding_str=encoding
    )
    print(cube)
    filepath = f"tmp_save_{str(encoding)}.nc"
    iris.save(cube, filepath)
    show_result(filepath)


# @pytest.mark.parametrize("ndim", [1, 2])
# def test_convert_bytes_to_strings(ndim: int):
#     if ndim == 1:
#         source = convert_strings_to_chararray(TEST_STRINGS, 16)
#     elif ndim == 2:
#         source = np.stack([
#             convert_strings_to_chararray(TEST_STRINGS, 16),
#             convert_strings_to_chararray(TEST_COORD_VALS, 16),
#         ])
#     else:
#         raise ValueError(f"Unexpected param ndim={ndim}.")
#     # convert the strings to bytes
#     result = convert_bytesarray_to_strings(source)
#     print(result)
