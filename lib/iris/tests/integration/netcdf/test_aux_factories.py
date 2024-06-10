# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for aux-factory-related loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import iris
from iris.tests import stock as stock


@tests.skip_data
class TestAtmosphereSigma(tests.IrisTest):
    def setUp(self):
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

    def test_save(self):
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename)
            self.assertCDL(filename)

    def test_save_load_loop(self):
        # Ensure that the AtmosphereSigmaFactory is automatically loaded
        # when loading the file.
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename)
            cube = iris.load_cube(filename, "air_potential_temperature")
            assert cube.coords("air_pressure")


@tests.skip_data
class TestHybridPressure(tests.IrisTest):
    def setUp(self):
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

    def test_save(self):
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(self.cube, filename)
            self.assertCDL(filename)

    def test_save_load_loop(self):
        # Tests an issue where the variable names in the formula
        # terms changed to the standard_names instead of the variable names
        # when loading a previously saved cube.
        with (
            self.temp_filename(suffix=".nc") as filename,
            self.temp_filename(suffix=".nc") as other_filename,
        ):
            iris.save(self.cube, filename)
            cube = iris.load_cube(filename, "air_potential_temperature")
            iris.save(cube, other_filename)
            other_cube = iris.load_cube(other_filename, "air_potential_temperature")
            self.assertEqual(cube, other_cube)


@tests.skip_data
class TestSaveMultipleAuxFactories(tests.IrisTest):
    def test_hybrid_height_and_pressure(self):
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
        with self.temp_filename(suffix=".nc") as filename:
            iris.save(cube, filename)
            self.assertCDL(filename)

    def test_shared_primary(self):
        cube = stock.realistic_4d()
        factory = iris.aux_factory.HybridHeightFactory(
            cube.coord("level_height"),
            cube.coord("sigma"),
            cube.coord("surface_altitude"),
        )
        factory.rename("another altitude")
        cube.add_aux_factory(factory)
        with (
            self.temp_filename(suffix=".nc") as filename,
            self.assertRaisesRegex(ValueError, "multiple aux factories"),
        ):
            iris.save(cube, filename)

    def test_hybrid_height_cubes(self):
        hh1 = stock.simple_4d_with_hybrid_height()
        hh1.attributes["cube"] = "hh1"
        hh2 = stock.simple_4d_with_hybrid_height()
        hh2.attributes["cube"] = "hh2"
        sa = hh2.coord("surface_altitude")
        sa.points = sa.points * 10
        with self.temp_filename(".nc") as fname:
            iris.save([hh1, hh2], fname)
            cubes = iris.load(fname, "air_temperature")
            cubes = sorted(cubes, key=lambda cube: cube.attributes["cube"])
            self.assertCML(cubes)

    def test_hybrid_height_cubes_on_dimension_coordinate(self):
        hh1 = stock.hybrid_height()
        hh2 = stock.hybrid_height()
        sa = hh2.coord("surface_altitude")
        sa.points = sa.points * 10
        emsg = "Unable to create dimensonless vertical coordinate."
        with (
            self.temp_filename(".nc") as fname,
            self.assertRaisesRegex(ValueError, emsg),
        ):
            iris.save([hh1, hh2], fname)


if __name__ == "__main__":
    tests.main()
