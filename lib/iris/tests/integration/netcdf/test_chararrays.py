import netCDF4 as nc
import numpy as np
import pytest

import iris

NX, N_STRLEN = 3, 64
TEST_STRINGS = ["Münster", "London", "Amsterdam"]
TEST_COORD_VALS = ["bun", "éclair", "sandwich"]


def convert_chararray(string_array_1d, maxlen, encoding="utf-8"):
    bbytes = [text.encode(encoding) for text in string_array_1d]
    pad = b"\0" * maxlen
    bbytes = [(x + pad)[:maxlen] for x in bbytes]
    chararray = np.array([[bb[i : i + 1] for i in range(maxlen)] for bb in bbytes])
    return chararray


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
        cube = iris.load_cube(filepath)
        print(cube)
        if iris.loading.LOAD_PROBLEMS._problems:
            print(iris.loading.LOAD_PROBLEMS)
            print(
                "\n".join(iris.loading.LOAD_PROBLEMS._problems[0].stack_trace.format())
            )
            iris.loading.LOAD_PROBLEMS._problems = []
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
def test_encodings(encoding):
    # small change
    print(f"\n=========\nTesting encoding: {encoding}")
    filepath = f"tmp_{str(encoding)}.nc"
    do_as = encoding
    if encoding != "utf-32":
        do_as = "utf-8"
    TEST_CHARARRAY = convert_chararray(TEST_STRINGS, N_STRLEN, encoding=do_as)
    TEST_COORDARRAY = convert_chararray(TEST_COORD_VALS, N_STRLEN, encoding=do_as)
    make_testfile(filepath, TEST_CHARARRAY, TEST_COORDARRAY, encoding_str=encoding)
    show_result(filepath)
