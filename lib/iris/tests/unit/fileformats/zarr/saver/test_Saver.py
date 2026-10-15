# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.zarr.saver.Saver`."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.fileformats.cf.saver import CFSaver
from iris.fileformats.zarr.saver import Saver


@pytest.fixture
def cube():
    cube = Cube(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        standard_name="air_temperature",
        units="K",
        var_name="air_temperature",
    )
    cube.add_dim_coord(
        DimCoord([0.0, 1.0], standard_name="latitude", units="degrees"), 0
    )
    cube.add_dim_coord(
        DimCoord([0.0, 1.0, 2.0], standard_name="longitude", units="degrees"), 1
    )
    return cube


class TestTheClass:
    def test_it_is_a_cf_saver(self):
        assert issubclass(Saver, CFSaver)

    def test_it_opens_a_store(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert sorted(group.array_keys()) == [
            "air_temperature",
            "latitude",
            "longitude",
        ]

    def test_the_default_mode_refuses_an_existing_store(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        with pytest.raises(FileExistsError):
            Saver(tmp_path / "out.zarr")

    def test_a_group_implies_adding(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        with Saver(tmp_path / "out.zarr", group="later") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert "later" in list(group.group_keys())
        assert "air_temperature" in list(group.array_keys())

    def test_an_explicit_mode_beats_the_group_default(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        with pytest.raises(FileExistsError):
            Saver(tmp_path / "out.zarr", mode="w-", group=None)


class TestStringVariables:
    def test_a_label_coordinate_is_a_string_array(self, tmp_path, cube):
        # netCDF stores text as bytes over an extra dimension. Zarr has a
        # string type, so the extra dimension would be an invention.
        cube.add_aux_coord(AuxCoord(np.array(["north", "south"]), long_name="place"), 0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        place = group["place"]
        assert list(place[:]) == ["north", "south"]
        assert place.metadata.dimension_names == ("latitude",)

    def test_no_string_dimension_is_invented(self, tmp_path, cube):
        cube.add_aux_coord(AuxCoord(np.array(["north", "south"]), long_name="place"), 0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        names = {
            name
            for array in group.arrays()
            for name in (array[1].metadata.dimension_names or ())
        }
        assert not {name for name in names if name.startswith("string")}


class TestScalarCubes:
    def test_a_scalar_cube_is_saved(self, tmp_path):
        # A scalar cube is ordinary, and saves to netCDF without complaint.
        cube = Cube(
            np.float32(3.0),
            standard_name="air_temperature",
            units="K",
            var_name="air_temperature",
        )
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert group["air_temperature"][()] == pytest.approx(3.0)


class TestGridMappings:
    def test_a_coordinate_system_is_written(self, tmp_path, cube):
        # `_grid_mapping_writer`'s base implementation, reached for real:
        # without it this raises AttributeError on `.variable`.
        from iris.coord_systems import GeogCS

        for coord in cube.coords():
            coord.coord_system = GeogCS(6371229.0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        grid = group["latitude_longitude"]
        assert grid.attrs["grid_mapping_name"] == "latitude_longitude"
        assert grid.attrs["earth_radius"] == pytest.approx(6371229.0)
