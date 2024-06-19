# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._concatenate.concatenate.py`."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import cf_units
import numpy as np
import numpy.ma as ma

from iris._concatenate import concatenate
from iris._lazy_data import as_lazy_data
from iris.aux_factory import HybridHeightFactory
import iris.coords
import iris.cube
from iris.exceptions import ConcatenateError


class TestEpoch(tests.IrisTest):
    def simple_1d_time_cubes(self, reftimes, coords_points):
        cubes = []
        data_points = [273, 275, 278, 277, 274]
        for reftime, coord_points in zip(reftimes, coords_points):
            cube = iris.cube.Cube(
                np.array(data_points, dtype=np.float32),
                standard_name="air_temperature",
                units="K",
            )
            unit = cf_units.Unit(reftime, calendar="standard")
            coord = iris.coords.DimCoord(
                points=np.array(coord_points, dtype=np.float32),
                standard_name="time",
                units=unit,
            )
            cube.add_dim_coord(coord, 0)
            cubes.append(cube)
        return cubes

    def test_concat_1d_with_same_time_units(self):
        reftimes = [
            "hours since 1970-01-01 00:00:00",
            "hours since 1970-01-01 00:00:00",
        ]
        coords_points = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        cubes = self.simple_1d_time_cubes(reftimes, coords_points)
        result = concatenate(cubes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10,))


class _MessagesMixin(tests.IrisTest):
    def setUp(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        # Time coord
        t_unit = cf_units.Unit("hours since 1970-01-01 00:00:00", calendar="standard")
        t_coord = iris.coords.DimCoord(
            points=np.arange(2, dtype=np.float32),
            standard_name="time",
            units=t_unit,
        )
        cube.add_dim_coord(t_coord, 0)
        # Lats and lons
        x_coord = iris.coords.DimCoord(
            points=np.arange(3, dtype=np.float32),
            standard_name="longitude",
            units="degrees",
        )
        cube.add_dim_coord(x_coord, 1)
        y_coord = iris.coords.DimCoord(
            points=np.arange(4, dtype=np.float32),
            standard_name="latitude",
            units="degrees",
        )
        cube.add_dim_coord(y_coord, 2)
        # Scalars
        cube.add_aux_coord(iris.coords.AuxCoord([0], "height", units="m"))
        # Aux Coords
        cube.add_aux_coord(
            iris.coords.AuxCoord(data, long_name="wibble", units="1"),
            data_dims=(0, 1, 2),
        )
        cube.add_aux_coord(
            iris.coords.AuxCoord([0, 1, 2], long_name="foo", units="1"),
            data_dims=(1,),
        )
        # Cell Measures
        cube.add_cell_measure(
            iris.coords.CellMeasure([0, 1, 2], long_name="bar", units="1"),
            data_dims=(1,),
        )
        # Ancillary Variables
        cube.add_ancillary_variable(
            iris.coords.AncillaryVariable([0, 1, 2], long_name="baz", units="1"),
            data_dims=(1,),
        )
        # Derived Coords
        delta = iris.coords.AuxCoord(0.0, var_name="delta", units="m")
        sigma = iris.coords.AuxCoord(1.0, var_name="sigma", units="1")
        orog = iris.coords.AuxCoord(2.0, var_name="orog", units="m")
        cube.add_aux_coord(delta, ())
        cube.add_aux_coord(sigma, ())
        cube.add_aux_coord(orog, ())
        cube.add_aux_factory(HybridHeightFactory(delta, sigma, orog))
        self.cube = cube


class TestMessages(_MessagesMixin):
    def setUp(self):
        super().setUp()

    def test_dim_coords_same_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        exc_regexp = "Cannot find an axis to concatenate over for phenomenon *"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_definition_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.units = "1"
        exc_regexp = "Cube metadata differs for phenomenon: *"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_dimensions_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord("latitude")
        exc_regexp = "Dimension coordinates differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_dimensions_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord("latitude").long_name = "bob"
        exc_regexp = "Dimension coordinates metadata differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_aux_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord("foo")
        exc_regexp = "Auxiliary coordinates differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_aux_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord("foo").units = "m"
        exc_regexp = "Auxiliary coordinates metadata differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_scalar_coords_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_coord("height")
        exc_regexp = "Scalar coordinates differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_scalar_coords_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord("height").long_name = "alice"
        exc_regexp = "Scalar coordinates values or metadata differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_cell_measure_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_cell_measure("bar")
        exc_regexp = "Cell measures differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_cell_measure_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.cell_measure("bar").units = "m"
        exc_regexp = "Cell measures metadata differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_ancillary_variable_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_ancillary_variable("baz")
        exc_regexp = "Ancillary variables differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_ancillary_variable_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.ancillary_variable("baz").units = "m"
        exc_regexp = "Ancillary variables metadata differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_derived_coord_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.remove_aux_factory(cube_2.aux_factories[0])
        exc_regexp = "Derived coordinates differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_derived_coord_metadata_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.aux_factories[0].units = "km"
        exc_regexp = "Derived coordinates metadata differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_ndim_difference_message(self):
        cube_1 = self.cube
        cube_2 = iris.cube.Cube(
            np.arange(5, dtype=np.float32),
            standard_name="air_temperature",
            units="K",
        )
        x_coord = iris.coords.DimCoord(
            points=np.arange(5, dtype=np.float32),
            standard_name="longitude",
            units="degrees",
        )
        cube_2.add_dim_coord(x_coord, 0)
        exc_regexp = "Data dimensions differ: [0-9] != [0-9]"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_datatype_difference_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.data.dtype = np.float64
        exc_regexp = "Data types differ: .* != .*"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)

    def test_dim_coords_overlap_message(self):
        cube_1 = self.cube
        cube_2 = cube_1.copy()
        cube_2.coord("time").points = np.arange(1, 3, dtype=np.float32)
        exc_regexp = "Found cubes with overlap on concatenate axis"
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([cube_1, cube_2], True)


class TestNonMetadataMessages(_MessagesMixin):
    def setUp(self):
        super().setUp()
        cube_2 = self.cube.copy()
        cube_2.coord("time").points = cube_2.coord("time").points + 2
        self.cube_2 = cube_2

    def test_aux_coords_diff_message(self):
        self.cube_2.coord("foo").points = [3, 4, 5]

        exc_regexp = "Auxiliary coordinates are unequal for phenomenon * "
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], True)
        with self.assertWarnsRegex(iris.warnings.IrisUserWarning, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], False)

    def test_cell_measures_diff_message(self):
        self.cube_2.cell_measure("bar").data = [3, 4, 5]

        exc_regexp = "Cell measures are unequal for phenomenon * "
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], True)
        with self.assertWarnsRegex(iris.warnings.IrisUserWarning, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], False)

    def test_ancillary_variable_diff_message(self):
        self.cube_2.ancillary_variable("baz").data = [3, 4, 5]

        exc_regexp = "Ancillary variables are unequal for phenomenon * "
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], True)
        with self.assertWarnsRegex(iris.warnings.IrisUserWarning, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], False)

    def test_derived_coords_diff_message(self):
        self.cube_2.aux_factories[0].update(self.cube_2.coord("sigma"), None)

        exc_regexp = "Derived coordinates are unequal for phenomenon * "
        with self.assertRaisesRegex(ConcatenateError, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], True)
        with self.assertWarnsRegex(iris.warnings.IrisUserWarning, exc_regexp):
            _ = concatenate([self.cube, self.cube_2], False)


class TestOrder(tests.IrisTest):
    def _make_cube(self, points, bounds=None):
        nx = 4
        data = np.arange(len(points) * nx).reshape(len(points), nx)
        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        lat = iris.coords.DimCoord(points, "latitude", bounds=bounds)
        lon = iris.coords.DimCoord(np.arange(nx), "longitude")
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        return cube

    def test_asc_points(self):
        top = self._make_cube([10, 30, 50, 70, 90])
        bottom = self._make_cube([-90, -70, -50, -30, -10])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_asc_bounds(self):
        top = self._make_cube([22.5, 67.5], [[0, 45], [45, 90]])
        bottom = self._make_cube([-67.5, -22.5], [[-90, -45], [-45, 0]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_asc_points_with_singleton_ordered(self):
        top = self._make_cube([5])
        bottom = self._make_cube([15, 25])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_asc_points_with_singleton_unordered(self):
        top = self._make_cube([25])
        bottom = self._make_cube([5, 15])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_asc_bounds_with_singleton_ordered(self):
        top = self._make_cube([5], [[0, 10]])
        bottom = self._make_cube([15, 25], [[10, 20], [20, 30]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_asc_bounds_with_singleton_unordered(self):
        top = self._make_cube([25], [[20, 30]])
        bottom = self._make_cube([5, 15], [[0, 10], [10, 20]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_points(self):
        top = self._make_cube([90, 70, 50, 30, 10])
        bottom = self._make_cube([-10, -30, -50, -70, -90])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_bounds(self):
        top = self._make_cube([67.5, 22.5], [[90, 45], [45, 0]])
        bottom = self._make_cube([-22.5, -67.5], [[0, -45], [-45, -90]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_points_with_singleton_ordered(self):
        top = self._make_cube([25])
        bottom = self._make_cube([15, 5])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_points_with_singleton_unordered(self):
        top = self._make_cube([5])
        bottom = self._make_cube([25, 15])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_bounds_with_singleton_ordered(self):
        top = self._make_cube([25], [[30, 20]])
        bottom = self._make_cube([15, 5], [[20, 10], [10, 0]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_desc_bounds_with_singleton_unordered(self):
        top = self._make_cube([5], [[10, 0]])
        bottom = self._make_cube([25, 15], [[30, 20], [20, 10]])
        result = concatenate([top, bottom])
        self.assertEqual(len(result), 1)

    def test_points_all_singleton(self):
        top = self._make_cube([5])
        bottom = self._make_cube([15])
        result1 = concatenate([top, bottom])
        result2 = concatenate([bottom, top])
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        self.assertEqual(result1, result2)

    def test_asc_bounds_all_singleton(self):
        top = self._make_cube([5], [0, 10])
        bottom = self._make_cube([15], [10, 20])
        result1 = concatenate([top, bottom])
        result2 = concatenate([bottom, top])
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        self.assertEqual(result1, result2)

    def test_desc_bounds_all_singleton(self):
        top = self._make_cube([5], [10, 0])
        bottom = self._make_cube([15], [20, 10])
        result1 = concatenate([top, bottom])
        result2 = concatenate([bottom, top])
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        self.assertEqual(result1, result2)


class TestConcatenate__dask(tests.IrisTest):
    def build_lazy_cube(self, points, bounds=None, nx=4, aux_coords=False):
        data = np.arange(len(points) * nx).reshape(len(points), nx)
        data = as_lazy_data(data)
        cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
        lat = iris.coords.DimCoord(points, "latitude", bounds=bounds)
        lon = iris.coords.DimCoord(np.arange(nx), "longitude")
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        if aux_coords:
            bounds = np.arange(len(points) * nx * 4).reshape(len(points), nx, 4)
            bounds = as_lazy_data(bounds)
            aux_coord = iris.coords.AuxCoord(data, var_name="aux_coord", bounds=bounds)
            cube.add_aux_coord(aux_coord, (0, 1))
        return cube

    def test_lazy_concatenate(self):
        c1 = self.build_lazy_cube([1, 2])
        c2 = self.build_lazy_cube([3, 4, 5])
        (cube,) = concatenate([c1, c2])
        self.assertTrue(cube.has_lazy_data())
        self.assertFalse(ma.isMaskedArray(cube.data))

    def test_lazy_concatenate_aux_coords(self):
        c1 = self.build_lazy_cube([1, 2], aux_coords=True)
        c2 = self.build_lazy_cube([3, 4, 5], aux_coords=True)
        (result,) = concatenate([c1, c2])

        self.assertTrue(c1.coord("aux_coord").has_lazy_points())
        self.assertTrue(c1.coord("aux_coord").has_lazy_bounds())

        self.assertTrue(c2.coord("aux_coord").has_lazy_points())
        self.assertTrue(c2.coord("aux_coord").has_lazy_bounds())

        self.assertTrue(result.coord("aux_coord").has_lazy_points())
        self.assertTrue(result.coord("aux_coord").has_lazy_bounds())

    def test_lazy_concatenate_masked_array_mixed_deferred(self):
        c1 = self.build_lazy_cube([1, 2])
        c2 = self.build_lazy_cube([3, 4, 5])
        c2.data = np.ma.masked_greater(c2.data, 3)
        (cube,) = concatenate([c1, c2])
        self.assertTrue(cube.has_lazy_data())
        self.assertTrue(ma.isMaskedArray(cube.data))


if __name__ == "__main__":
    tests.main()
