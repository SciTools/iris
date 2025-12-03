# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Temporary code to confirm how various numpy dtypes are stored in a netcdf file."""

import netCDF4 as nc
import numpy as np
import pytest

from iris.tests.integration.netcdf.test_chararrays import ncdump

# types = [
#     "i1",  # np.int8
#     "u1",  # np.uint8
#     "S1",  # np.byte_
#     "U1",  # np.str_
#     "S",  # multibytes
#     "U",  # unicode strings, with/without non-ascii content
# ]

samples = {
    "i1": [-5, 7, 35],  # np.int8
    "u1": [65, 67, 90],  # np.uint8
    "S1": [b"A", b"B", b"Z"],  # np.byte_
    "U1": ["A", "B", "C"],  # np.str_
    "S": [b"one21", b"three", b""],  # multibyte
    "U": ["one", "éclair", "nine"],  # unicode strings
}
sample_arrays = {
    type_code: np.array(values, dtype=type_code)
    for type_code, values in samples.items()
}


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    return tmp_path_factory.mktemp("netcdf")


def create_file(array: np.ndarray, path):
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("x", 3)
        v = ds.createVariable("vx", array.dtype, ("x",))
        # v.set_auto_chartostring(False)
        v._Encoding = "UTF-8" if array.dtype.kind == "U" else "ascii"
        v[:] = array


def get_loadback_array(path):
    with nc.Dataset(str(path), "r") as ds:
        v = ds.variables["vx"]
        v.set_auto_chartostring(False)
        result = v[:]
    return result


@pytest.mark.parametrize("dtype", list(samples.keys()))
def test(tmpdir, dtype):
    arr = sample_arrays[dtype]
    print("\n---")
    print(dtype)
    path = tmpdir / f"tmp_{dtype}.nc"
    create_file(arr, path)
    ncdump(path, "-s")
    loadback_array = get_loadback_array(path)
    print(f"  SPEC:{dtype} SAVED-AS:{arr.dtype} RELOAD-AS:{loadback_array.dtype}")


# from iris.tests import env_bin_path
# NCGEN_PATHSTR = str(env_bin_path("ncgen"))
#
#
# def ncgen(cdl_path, nc_path, *args):
#     """Call ncdump to print a dump of a file."""
#     args = list(args)
#     if not any(arg.startswith('-k') for arg in args):
#         args[:0] = ["-k", "nc4"]  # force netcdf4
#     call_args = [NCGEN_PATHSTR] + list(args) + [str(cdl_path), '-o', str(nc_path)]
#     subprocess.check_call(call_args)
#
#
# def test_uchar(tmpdir):
#     arr = sample_arrays["S1"]
#     path = tmpdir / f"tmp_ichar.nc"
#     create_file(arr, path)
#     text = ncdump(path, "-s")
#     text_u = text.replace("\t", "   ")
#     text_u = text_u.replace(" char ", " unsigned char ")
#     cdl_path = tmpdir / f"tmp_uchar.cdl"
#     with open(cdl_path, "w") as f_out:
#         f_out.write(text_u)
#     nc_path_2 = tmpdir / f"tmp_uchar.nc"
#     ncgen(cdl_path, nc_path_2)
#     loadback_array = get_loadback_array(nc_path_2)
#     print(f"  netcdf type 'uchar' LOADS-AS:{loadback_array.dtype}")
