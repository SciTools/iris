# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for save+load of datales cubes."""

import numpy as np
import pytest

import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper
from iris.fileformats.netcdf.saver import Saver


class TestDataless:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path_factory):
        ny, nx = 3, 4
        self.testcube = Cube(
            shape=(ny, nx),
            long_name="testdata",
            dim_coords_and_dims=[
                (DimCoord(np.arange(ny), long_name="y"), 0),
                (DimCoord(np.arange(nx), long_name="x"), 1),
            ],
        )
        self.testdir = tmp_path_factory.mktemp("dataless")
        self.test_path = self.testdir / "test.nc"

    @staticmethod
    def _strip_saveload_additions(reloaded_cube):
        reloaded_cube.attributes.pop("Conventions", None)
        reloaded_cube.var_name = None
        for co in reloaded_cube.coords():
            co.var_name = None

    def test_dataless_save(self):
        # Check that we can save a dataless cube, and what that looks like in the file.
        iris.save(self.testcube, self.test_path)
        assert Saver._DATALESS_ATTRNAME not in self.testcube.attributes
        # Check the content as seen in the file
        ncds = DatasetWrapper(self.test_path)
        var = ncds.variables["testdata"]
        assert Saver._DATALESS_ATTRNAME in var.ncattrs()
        assert var.dtype == Saver._DATALESS_DTYPE
        assert "_FillValue" in var.ncattrs()
        assert var._FillValue == Saver._DATALESS_FILLVALUE
        assert np.all(np.ma.getmaskarray(var[:]) == True)  # noqa: E712

    def test_dataless_load(self):
        # Check that we can load a saved dataless cube, and it matches the original.
        iris.save(self.testcube, self.test_path)

        # NB Load with load_raw, since we haven't finished supporting dataless merge.
        (result_cube,) = iris.load_raw(self.test_path)
        assert result_cube.is_dataless()
        assert "iris_dataless_cube" not in result_cube.attributes

        # strip off extra things added by netcdf save+load
        self._strip_saveload_additions(result_cube)

        # Result now == original
        assert result_cube == self.testcube

    def test_mixture_saveload(self):
        # Check that a mixture of dataless and "normal" cubes can be saved + loaded back
        dataless = self.testcube
        ny = dataless.shape[0]
        dataful = Cube(
            np.ones((ny, 3)),
            long_name="other",
            dim_coords_and_dims=[(dataless.coord("y"), 0)],
        )
        iris.save([dataless, dataful], self.test_path)
        # NB Load with load_raw, since we haven't finished supporting dataless merge.
        cubes = iris.load_raw(self.test_path)
        assert len(cubes) == 2
        read_dataless = cubes.extract_cube("testdata")
        read_dataful = cubes.extract_cube("other")
        assert read_dataless.is_dataless()
        assert not read_dataful.is_dataless()
        for cube in (read_dataless, read_dataful):
            self._strip_saveload_additions(cube)
        assert read_dataless == dataless
        assert read_dataful == dataful

    def test_nodata_size(self):
        # Check that a file saved with a large dataless cube does *not* occupy a large
        # amount of diskspace.
        ny, nx = 10000, 10000
        data_dims = (ny, nx)
        dataless_cube = Cube(shape=data_dims)

        iris.save(dataless_cube, self.test_path)

        data_size_bytes = ny * nx  # bytes, since dtype is "u1" (approx 100Mb)
        filesize_bytes = self.test_path.stat().st_size
        # Check that the file size < 1/10 variable array size
        # The 0.1 is a bit arbitrary, but it makes the point!
        assert filesize_bytes < 0.1 * data_size_bytes
