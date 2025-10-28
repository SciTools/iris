import netCDF4 as nc
import numpy as np
import pytest

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

NX, N_STRLEN = 3, 64
TEST_STRINGS = ["Münster", "London", "Amsterdam"]
TEST_COORD_VALS = ["bun", "éclair", "sandwich"]


def convert_strings_to_chararray(string_array_1d, maxlen, encoding="utf-8"):
    bbytes = [text.encode(encoding) for text in string_array_1d]
    pad = b"\0" * maxlen
    bbytes = [(x + pad)[:maxlen] for x in bbytes]
    chararray = np.array([[bb[i : i + 1] for i in range(maxlen)] for bb in bbytes])
    return chararray


# def convert_chararray_to_strings(char_array_2d, maxlen: int | None =0, encoding="utf-8"):
#     strings = [bytes.decode(encoding) for bytes in char_array_2d]
#     if not maxlen:
#         maxlen = max(len(string) for string in strings)
#     dtype_str = f"S{maxlen}"
#     string_array = np.array(strings, dtype=dtype_str)
#     return string_array


INCLUDE_COORD = True
# INCLUDE_COORD = False


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
            v.coordinates = "v_co"


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
        if INCLUDE_COORD:
            print("-coord data-")
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
