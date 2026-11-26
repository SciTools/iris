# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""What the netCDF saver does with a dtype the target format cannot hold.

Relocating the CF saver for Zarr support (SciTools/iris#6977) moved the
narrowing decision out of the saver, which now asks the dataset instead of
reading netCDF format strings. Every answer here survived that move except
one: SciTools/iris#7337 was fixed in the same change, so "NETCDF3_64BIT"
narrows now where it used to fail inside the netCDF C library.
"""

import numpy as np
import pytest

import iris
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.fileformats.netcdf._dataset import NetCDFDataset

#: Every format Saver accepts.
ALL_FORMATS = ("NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_CLASSIC", "NETCDF3_64BIT")


def saved_dtype(cube, path, netcdf_format):
    """Save the cube, and return the dtype its data variable has in the file."""
    iris.save(cube, path, netcdf_format=netcdf_format)
    with NetCDFDataset(path) as dataset:
        return dataset.variables[cube.var_name].dtype


@pytest.fixture
def int64_cube():
    """Return an int64 cube whose values all fit in an int32."""
    cube = Cube(np.arange(4, dtype=np.int64), var_name="payload")
    cube.add_aux_coord(AuxCoord(np.arange(4, dtype=np.int64), var_name="index"), 0)
    return cube


class TestInt64:
    """int64 is the dtype the restricted formats cannot hold."""

    def test_netcdf4_keeps_it(self, int64_cube, tmp_path):
        assert saved_dtype(int64_cube, tmp_path / "out.nc", "NETCDF4") == np.int64

    @pytest.mark.parametrize(
        "netcdf_format", ["NETCDF4_CLASSIC", "NETCDF3_CLASSIC", "NETCDF3_64BIT"]
    )
    def test_the_restricted_formats_narrow_it(
        self, int64_cube, tmp_path, netcdf_format
    ):
        assert saved_dtype(int64_cube, tmp_path / "out.nc", netcdf_format) == np.int32

    @pytest.mark.parametrize(
        ("netcdf_format", "named"),
        [
            ("NETCDF4_CLASSIC", "NETCDF4_CLASSIC"),
            ("NETCDF3_CLASSIC", "NETCDF3_CLASSIC"),
            # The message names the format netCDF4 reports, which for this one
            # is not the format the caller asked for: SciTools/iris#7337.
            ("NETCDF3_64BIT", "NETCDF3_64BIT_OFFSET"),
        ],
    )
    def test_a_value_too_large_to_narrow_names_the_format(
        self, tmp_path, netcdf_format, named
    ):
        cube = Cube(np.array([2**40], dtype=np.int64), var_name="payload")
        with pytest.raises(ValueError, match=named):
            iris.save(cube, tmp_path / "out.nc", netcdf_format=netcdf_format)


class TestDtypesEveryFormatHolds:
    """A dtype the format holds is written as it stands, never narrowed."""

    @pytest.mark.parametrize("dtype", ["f4", "f8", "i2", "i4"])
    @pytest.mark.parametrize("netcdf_format", ALL_FORMATS)
    def test_not_narrowed(self, tmp_path, dtype, netcdf_format):
        cube = Cube(np.arange(4, dtype=dtype), var_name="payload")
        assert saved_dtype(cube, tmp_path / "out.nc", netcdf_format) == np.dtype(dtype)
