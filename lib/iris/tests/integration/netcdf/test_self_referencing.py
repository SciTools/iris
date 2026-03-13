# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for iris#3367 - loading a self-referencing NetCDF file."""

import numpy as np
import pytest

import iris
from iris.fileformats.netcdf import _thread_safe_nc
from iris.tests import _shared_utils
from iris.warnings import IrisCfMissingVarWarning


@_shared_utils.skip_data
class TestCMIP6VolcelloLoad:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.fname = _shared_utils.get_data_path(
            (
                "NetCDF",
                "volcello",
                "volcello_Ofx_CESM2_deforest-globe_r1i1p1f1_gn.nc",
            )
        )

    def test_cmip6_volcello_load_issue_3367(self):
        # Ensure that reading a file which references itself in
        # `cell_measures` can be read. At the same time, ensure that we
        # still receive a warning about other variables mentioned in
        # `cell_measures` i.e. a warning should be raised about missing
        # areacello.
        areacello_str = "areacello"
        volcello_str = "volcello"
        expected_msg = (
            "Missing CF-netCDF measure variable %r, "
            "referenced by netCDF variable %r" % (areacello_str, volcello_str)
        )

        with pytest.warns(IrisCfMissingVarWarning, match=expected_msg):
            # ensure file loads without failure
            cube = iris.load_cube(self.fname)

        # extra check to ensure correct variable was found
        assert cube.standard_name == "ocean_volume"


class TestSelfReferencingVarLoad:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.temp_dir_path = tmp_path / "issue_3367_volcello_test_file.nc"
        dataset = _thread_safe_nc.DatasetWrapper(self.temp_dir_path, "w")

        dataset.createDimension("lat", 4)
        dataset.createDimension("lon", 5)
        dataset.createDimension("lev", 3)

        latitudes = dataset.createVariable("lat", np.float64, ("lat",))
        longitudes = dataset.createVariable("lon", np.float64, ("lon",))
        levels = dataset.createVariable("lev", np.float64, ("lev",))
        volcello = dataset.createVariable("volcello", np.float32, ("lat", "lon", "lev"))

        latitudes.standard_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.axis = "Y"
        latitudes[:] = np.linspace(-90, 90, 4)

        longitudes.standard_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.axis = "X"
        longitudes[:] = np.linspace(0, 360, 5)

        levels.standard_name = "olevel"
        levels.units = "centimeters"
        levels.positive = "down"
        levels.axis = "Z"
        levels[:] = np.linspace(0, 10**5, 3)

        volcello.id = "volcello"
        volcello.out_name = "volcello"
        volcello.standard_name = "ocean_volume"
        volcello.units = "m3"
        volcello.realm = "ocean"
        volcello.frequency = "fx"
        volcello.cell_measures = "area: areacello volume: volcello"
        volcello = np.arange(4 * 5 * 3).reshape((4, 5, 3))

        dataset.close()

    def test_self_referencing_load_issue_3367(self):
        # Ensure that reading a file which references itself in
        # `cell_measures` can be read. At the same time, ensure that we
        # still receive a warning about other variables mentioned in
        # `cell_measures` i.e. a warning should be raised about missing
        # areacello.
        areacello_str = "areacello"
        volcello_str = "volcello"
        expected_msg = (
            "Missing CF-netCDF measure variable %r, "
            "referenced by netCDF variable %r" % (areacello_str, volcello_str)
        )

        with pytest.warns(IrisCfMissingVarWarning, match=expected_msg):
            # ensure file loads without failure
            cube = iris.load_cube(self.temp_dir_path)

        # extra check to ensure correct variable was found
        assert cube.standard_name == "ocean_volume"
