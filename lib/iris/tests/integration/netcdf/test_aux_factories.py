# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for aux-factory-related loading and saving netcdf files."""

import pytest

import iris
from iris.tests import _shared_utils
from iris.tests import stock as stock


@_shared_utils.skip_data
class TestAtmosphereSigma:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Modify stock cube so it is suitable to have a atmosphere sigma
        # factory added to it.
        cube = stock.realistic_4d_no_derived()
        cube.coord("surface_altitude").rename("surface_air_pressure")
        cube.coord("surface_air_pressure").units = "Pa"
        cube.coord("sigma").units = "1"
        ptop_coord = iris.coords.AuxCoord(1000.0, var_name="ptop", units="Pa")
        cube.add_aux_coord(ptop_coord, ())
        cube.remove_coord("level_height")
        # Construct and add atmosphere sigma factory.
        factory = iris.aux_factory.AtmosphereSigmaFactory(
            cube.coord("ptop"),
            cube.coord("sigma"),
            cube.coord("surface_air_pressure"),
        )
        cube.add_aux_factory(factory)
        self.cube = cube

    def test_save(self, request, tmp_path):
        filename = tmp_path / "fn.nc"
        iris.save(self.cube, filename)
        _shared_utils.assert_CDL(request, filename)

    def test_save_load_loop(self, tmp_path):
        # Ensure that the AtmosphereSigmaFactory is automatically loaded
        # when loading the file.
        filename = tmp_path / "fn.nc"
        iris.save(self.cube, filename)
        cube = iris.load_cube(filename, "air_potential_temperature")
        assert cube.coords("air_pressure")


@_shared_utils.skip_data
class TestHybridPressure:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Modify stock cube so it is suitable to have a
        # hybrid pressure factory added to it.
        cube = stock.realistic_4d_no_derived()
        cube.coord("surface_altitude").rename("surface_air_pressure")
        cube.coord("surface_air_pressure").units = "Pa"
        cube.coord("level_height").rename("level_pressure")
        cube.coord("level_pressure").units = "Pa"
        # Construct and add hybrid pressure factory.
        factory = iris.aux_factory.HybridPressureFactory(
            cube.coord("level_pressure"),
            cube.coord("sigma"),
            cube.coord("surface_air_pressure"),
        )
        cube.add_aux_factory(factory)
        self.cube = cube

    def test_save(self, request, tmp_path):
        filename = tmp_path / "fn.nc"
        iris.save(self.cube, filename)
        _shared_utils.assert_CDL(request, filename)

    def test_save_load_loop(self, tmp_path):
        # Tests an issue where the variable names in the formula
        # terms changed to the standard_names instead of the variable names
        # when loading a previously saved cube.
        filename = tmp_path / "fn.nc"
        other_filename = tmp_path / "ofn.nc"
        iris.save(self.cube, filename)
        cube = iris.load_cube(filename, "air_potential_temperature")
        iris.save(cube, other_filename)
        other_cube = iris.load_cube(other_filename, "air_potential_temperature")
        assert cube == other_cube


@_shared_utils.skip_data
class TestSaveMultipleAuxFactories:
    def test_hybrid_height_and_pressure(self, request, tmp_path):
        cube = stock.realistic_4d()
        cube.add_aux_coord(
            iris.coords.DimCoord(1200.0, long_name="level_pressure", units="hPa")
        )
        cube.add_aux_coord(
            iris.coords.DimCoord(0.5, long_name="other sigma", units="1")
        )
        cube.add_aux_coord(
            iris.coords.DimCoord(1000.0, long_name="surface_air_pressure", units="hPa")
        )
        factory = iris.aux_factory.HybridPressureFactory(
            cube.coord("level_pressure"),
            cube.coord("other sigma"),
            cube.coord("surface_air_pressure"),
        )
        cube.add_aux_factory(factory)
        filename = tmp_path / "fn.nc"
        iris.save(cube, filename)
        _shared_utils.assert_CDL(request, filename)

    def test_shared_primary(self, tmp_path):
        cube = stock.realistic_4d()
        factory = iris.aux_factory.HybridHeightFactory(
            cube.coord("level_height"),
            cube.coord("sigma"),
            cube.coord("surface_altitude"),
        )
        factory.rename("another altitude")
        cube.add_aux_factory(factory)
        filename = tmp_path / "fn.nc"
        with pytest.raises(ValueError, match="multiple aux factories"):
            iris.save(cube, filename)

    def test_hybrid_height_cubes(self, request, tmp_path):
        hh1 = stock.simple_4d_with_hybrid_height()
        hh1.attributes["cube"] = "hh1"
        hh2 = stock.simple_4d_with_hybrid_height()
        hh2.attributes["cube"] = "hh2"
        sa = hh2.coord("surface_altitude")
        sa.points = sa.points * 10
        filename = tmp_path / "fn.nc"
        iris.save([hh1, hh2], filename)
        cubes = iris.load(filename, "air_temperature")
        cubes = sorted(cubes, key=lambda cube: cube.attributes["cube"])
        _shared_utils.assert_CML(request, cubes)

    def test_hybrid_height_cubes_on_dimension_coordinate(self, tmp_path):
        hh1 = stock.hybrid_height()
        hh2 = stock.hybrid_height()
        sa = hh2.coord("surface_altitude")
        sa.points = sa.points * 10
        emsg = "Unable to create dimensonless vertical coordinate."
        filename = tmp_path / "fn.nc"
        with pytest.raises(ValueError, match=emsg):
            iris.save([hh1, hh2], filename)
