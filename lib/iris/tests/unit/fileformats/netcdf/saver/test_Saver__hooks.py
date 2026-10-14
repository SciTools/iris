# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The netCDF-specific parts of the save chain, named and tested.

Relocating the CF saver (SciTools/iris#6977) moves the rest of the class to a
format-agnostic home. Each test here covers one of the parts that stays
behind, from the outside, so that the move cannot quietly change what a
backend is asked to do.
"""

import dask.array as da
import numpy as np
import pytest

import iris
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable
from iris.fileformats.netcdf._dataset import NetCDFDataset


class TestStringVariables:
    """Unicode data becomes a char variable over a string dimension."""

    @pytest.fixture
    def labelled_cube(self):
        cube = Cube(np.arange(3.0), var_name="payload")
        cube.add_aux_coord(
            AuxCoord(np.array(["alpha", "beta", "gamma"]), var_name="label"), 0
        )
        return cube

    def test_a_string_dimension_is_created_for_the_label(self, labelled_cube, tmp_path):
        path = tmp_path / "labelled.nc"
        iris.save(labelled_cube, path)
        with NetCDFDataset(path) as dataset:
            # Five ascii characters, so five bytes. The dimension is the
            # width, and exists only because netCDF has no unicode type.
            assert dataset.dimensions["string5"] == 5

    def test_the_label_round_trips(self, labelled_cube, tmp_path):
        path = tmp_path / "labelled.nc"
        iris.save(labelled_cube, path)
        loaded = iris.load_cube(path)
        assert list(loaded.coord("label").points) == ["alpha", "beta", "gamma"]


class TestNCZarrWrites:
    """NCZarr cannot be reopened, so lazy data is written before the close."""

    def test_lazy_data_reaches_an_nczarr_target(self, tmp_path):
        cube = Cube(da.arange(6.0, chunks=3).reshape(2, 3), long_name="payload")
        assert cube.has_lazy_data()
        uri = (tmp_path / "out.zarr").as_uri() + "#mode=nczarr,file"
        with iris.FUTURE.context(save_split_attrs=True):
            iris.save(cube, uri)
        loaded = iris.load_cube(uri)
        assert loaded.data.ravel().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


class TestCFPatch:
    """The site-configuration hook is documented as receiving netCDF4 objects."""

    def test_cf_patch_receives_raw_netcdf4_objects(self, mocker, tmp_path):
        seen = {}

        def record(profile, dataset, variable):
            seen["dataset"] = dataset
            seen["variable"] = variable

        mocker.patch.dict(
            iris.site_configuration,
            {"cf_profile": lambda cube: "profiled", "cf_patch": record},
        )
        iris.save(Cube(np.arange(3.0), var_name="payload"), tmp_path / "patched.nc")

        # Unwrapped, both of them. A CFVariable defines no __setattr__, so a
        # hook doing 'variable.name = value' against one would succeed and
        # write nothing at all.
        assert not isinstance(seen["dataset"], CFDataset)
        assert not isinstance(seen["variable"], CFDatasetVariable)
        assert hasattr(seen["dataset"], "createVariable")
        assert hasattr(seen["variable"], "setncattr")
