# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :meth:`iris.fileformats.cf.saver._CFSaver._grid_mapping_writer`.

The hook exists because netCDF4's own object is what has always received the
CF grid-mapping parameters. Changing where they land would change what is in a
netCDF file, which Q8 of the design document rules out of this programme, so
netCDF keeps writing onto that object and every other store writes into the
``attributes`` mapping.
"""

import types

import numpy as np
import pytest

import iris
from iris.coord_systems import GeogCS
from iris.cube import Cube
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


def fake_cf_var():
    """Return a variable whose two write destinations can be told apart.

    A real variable's ``attributes`` is a view onto the netCDF4 object, so
    writing to either one shows up in both, and the tests below could not say
    which route a parameter took.
    """
    return types.SimpleNamespace(attributes={}, variable=types.SimpleNamespace())


class TestTheBaseWriter:
    def test_it_writes_into_the_attributes_mapping(self, tmp_path):
        from iris.fileformats.zarr.saver import Saver

        cf_var = fake_cf_var()
        with Saver(tmp_path / "out.zarr") as saver:
            write_parameter = saver._grid_mapping_writer(cf_var)
        write_parameter("earth_radius", 6371229.0)
        write_parameter("horizontal_datum_name", "OSGB 1936")

        assert cf_var.attributes == {
            "earth_radius": 6371229.0,
            "horizontal_datum_name": "OSGB 1936",
        }


class TestTheNetCDFOverride:
    def test_it_writes_onto_the_netcdf4_variable(self, tmp_path):
        # Not just "writes the same values": it must bypass `.attributes`,
        # whose ASCII-to-bytes coercion would change how netCDF types each
        # parameter in the file.
        from iris.fileformats.netcdf.saver import Saver

        cf_var = fake_cf_var()
        with Saver(tmp_path / "out.nc", "NETCDF4") as saver:
            write_parameter = saver._grid_mapping_writer(cf_var)
        write_parameter("crs_wkt", "GEOGCRS[...]")

        assert cf_var.variable.crs_wkt == "GEOGCRS[...]"
        assert cf_var.attributes == {}


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
