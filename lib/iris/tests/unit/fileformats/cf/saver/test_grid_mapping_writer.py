# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :meth:`iris.fileformats.cf.saver.CFSaver._grid_mapping_writer`.

The hook exists because the 63 grid-mapping parameters are written by plain
attribute assignment, and netCDF4's own object is what receives them today.
Changing where they land would change what is in a netCDF file, which Q8 of
the design document rules out of this programme, so netCDF keeps that object
and everything else gets a writer that uses the ``attributes`` mapping.
"""

import numpy as np
import pytest

import iris
from iris.coord_systems import GeogCS
from iris.cube import Cube
from iris.fileformats.cf.saver import CFSaver, _GridMappingAttributeWriter
from iris.fileformats.netcdf._thread_safe_nc import DatasetWrapper


@pytest.fixture
def cube():
    cube = Cube(np.arange(6, dtype=np.float32).reshape(2, 3), var_name="air")
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.array([0.0, 1.0]),
            standard_name="latitude",
            units="degrees",
            coord_system=GeogCS(6371229.0),
        ),
        0,
    )
    cube.add_dim_coord(
        iris.coords.DimCoord(
            np.array([0.0, 1.0, 2.0]),
            standard_name="longitude",
            units="degrees",
            coord_system=GeogCS(6371229.0),
        ),
        1,
    )
    return cube


class TestTheWriterItself:
    def test_it_writes_through_the_attributes_mapping(self):
        recorded = {}

        class FakeVariable:
            attributes = recorded

        writer = _GridMappingAttributeWriter(FakeVariable())
        writer.earth_radius = 6371229.0
        writer.horizontal_datum_name = "OSGB 1936"

        assert recorded == {
            "earth_radius": 6371229.0,
            "horizontal_datum_name": "OSGB 1936",
        }

    def test_it_does_not_grow_members_of_its_own(self):
        # Every assignment must reach the variable. An attribute that stuck to
        # the writer instead would be silently dropped from the file.
        class FakeVariable:
            attributes: dict = {}

        writer = _GridMappingAttributeWriter(FakeVariable())
        writer.anything = 1
        assert "anything" not in vars(writer)


class TestTheNetCDFOverride:
    def test_it_is_the_netcdf4_variable_itself(self, tmp_path, cube):
        # Not "writes the same values": the same object, so the grid-mapping
        # assignments take the path they always have and the file cannot change.
        from iris.fileformats.netcdf.saver import Saver

        with Saver(tmp_path / "out.nc", "NETCDF4") as saver:
            cf_var = saver._dataset.create_variable("probe", np.int32)
            assert saver._grid_mapping_writer(cf_var) is cf_var.variable


class TestTheFileIsUnchanged:
    def test_grid_mapping_attributes_land_where_they_did(self, tmp_path, cube):
        path = tmp_path / "out.nc"
        iris.save(cube, path)

        dataset = DatasetWrapper(path)
        grid = dataset.variables["latitude_longitude"]
        assert grid.getncattr("grid_mapping_name") == "latitude_longitude"
        assert grid.getncattr("earth_radius") == pytest.approx(6371229.0)
        assert grid.getncattr("longitude_of_prime_meridian") == pytest.approx(0.0)
        dataset.close()
