# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris._concatenate.concatenate.py`."""

import cf_units
import numpy as np
import numpy.ma as ma
import pytest

from iris._concatenate import concatenate
from iris._lazy_data import as_lazy_data
from iris.aux_factory import HybridHeightFactory
import iris.coords
import iris.cube
from iris.exceptions import ConcatenateError
import iris.warnings


class TestEpoch:
    @pytest.fixture()
    def simple_1d_time_cubes(self):
        reftimes = [
            "hours since 1970-01-01 00:00:00",
            "hours since 1970-01-01 00:00:00",
        ]
        coords_points = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        data_points = [273, 275, 278, 277, 274]
        cubes = []
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

    def test_concat_1d_with_same_time_units(self, simple_1d_time_cubes):
        result = concatenate(simple_1d_time_cubes)
        assert len(result) == 1
        assert result[0].shape == (10,)


class _MessagesMixin:
    @pytest.fixture()
    def placeholder(self):
        # Shim to allow sample_cubes to have identical signature in both parent and subclasses
        return []

    @pytest.fixture()
    def sample_cubes(self, placeholder):
        # Construct and return a pair of identical cubes
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
        # Return a list with two identical cubes
        return [cube, cube.copy()]

    def test_definition_difference_message(self, sample_cubes):
        sample_cubes[1].units = "1"
        exc_regexp = "Cube metadata differs for phenomenon:"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)


class TestMessages(_MessagesMixin):
    def test_dim_coords_same_message(self, sample_cubes):
        exc_regexp = "Cannot find an axis to concatenate over for phenomenon *"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_definition_difference_message(self, sample_cubes):
        sample_cubes[1].units = "1"
        exc_regexp = "Cube metadata differs for phenomenon: *"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_dimensions_difference_message(self, sample_cubes):
        sample_cubes[1].remove_coord("latitude")
        exc_regexp = "Dimension coordinates differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_dimensions_metadata_difference_message(self, sample_cubes):
        sample_cubes[1].coord("latitude").long_name = "bob"
        exc_regexp = "Dimension coordinates metadata differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_aux_coords_difference_message(self, sample_cubes):
        sample_cubes[1].remove_coord("foo")
        exc_regexp = "Auxiliary coordinates differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_aux_coords_metadata_difference_message(self, sample_cubes):
        sample_cubes[1].coord("foo").units = "m"
        exc_regexp = "Auxiliary coordinates metadata differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_scalar_coords_difference_message(self, sample_cubes):
        sample_cubes[1].remove_coord("height")
        exc_regexp = "Scalar coordinates differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_scalar_coords_metadata_difference_message(self, sample_cubes):
        sample_cubes[1].coord("height").long_name = "alice"
        exc_regexp = "Scalar coordinates values or metadata differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_cell_measure_difference_message(self, sample_cubes):
        sample_cubes[1].remove_cell_measure("bar")
        exc_regexp = "Cell measures differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_cell_measure_metadata_difference_message(self, sample_cubes):
        sample_cubes[1].cell_measure("bar").units = "m"
        exc_regexp = "Cell measures metadata differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_ancillary_variable_difference_message(self, sample_cubes):
        sample_cubes[1].remove_ancillary_variable("baz")
        exc_regexp = "Ancillary variables differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_ancillary_variable_metadata_difference_message(self, sample_cubes):
        sample_cubes[1].ancillary_variable("baz").units = "m"
        exc_regexp = "Ancillary variables metadata differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_derived_coord_difference_message(self, sample_cubes):
        sample_cubes[1].remove_aux_factory(sample_cubes[1].aux_factories[0])
        exc_regexp = "Derived coordinates differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_derived_coord_metadata_difference_message(self, sample_cubes):
        sample_cubes[1].aux_factories[0].units = "km"
        exc_regexp = "Derived coordinates metadata differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_ndim_difference_message(self, sample_cubes):
        # Replace cube#2 with an entirely different thing
        sample_cubes[1] = iris.cube.Cube(
            np.arange(5, dtype=np.float32),
            standard_name="air_temperature",
            units="K",
        )
        x_coord = iris.coords.DimCoord(
            points=np.arange(5, dtype=np.float32),
            standard_name="longitude",
            units="degrees",
        )
        sample_cubes[1].add_dim_coord(x_coord, 0)
        exc_regexp = "Data dimensions differ: [0-9] != [0-9]"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_datatype_difference_message(self, sample_cubes):
        sample_cubes[1].data.dtype = np.float64
        exc_regexp = "Data types differ: .* != .*"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)

    def test_dim_coords_overlap_message(self, sample_cubes):
        sample_cubes[1].coord("time").points = np.arange(1, 3, dtype=np.float32)
        exc_regexp = "Found cubes with overlap on concatenate axis"
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)


class TestNonMetadataMessages(_MessagesMixin):
    parent_cubes = _MessagesMixin.sample_cubes

    @pytest.fixture()
    def sample_cubes(self, parent_cubes):
        coord = parent_cubes[1].coord("time")
        parent_cubes[1].replace_coord(coord.copy(points=coord.points + 2))
        return parent_cubes

    def test_aux_coords_diff_message(self, sample_cubes):
        sample_cubes[1].coord("foo").points = [3, 4, 5]

        exc_regexp = "Auxiliary coordinates are unequal for phenomenon * "
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)
        with pytest.warns(iris.warnings.IrisUserWarning, match=exc_regexp):
            _ = concatenate(sample_cubes, False)

    def test_cell_measures_diff_message(self, sample_cubes):
        sample_cubes[1].cell_measure("bar").data = [3, 4, 5]

        exc_regexp = "Cell measures are unequal for phenomenon * "
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)
        with pytest.warns(iris.warnings.IrisUserWarning, match=exc_regexp):
            _ = concatenate(sample_cubes, False)

    def test_ancillary_variable_diff_message(self, sample_cubes):
        sample_cubes[1].ancillary_variable("baz").data = [3, 4, 5]

        exc_regexp = "Ancillary variables are unequal for phenomenon * "
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)
        with pytest.warns(iris.warnings.IrisUserWarning, match=exc_regexp):
            _ = concatenate(sample_cubes, False)

    def test_derived_coords_diff_message(self, sample_cubes):
        sample_cubes[1].aux_factories[0].update(sample_cubes[1].coord("sigma"), None)

        exc_regexp = "Derived coordinates are unequal for phenomenon * "
        with pytest.raises(ConcatenateError, match=exc_regexp):
            _ = concatenate(sample_cubes, True)
        with pytest.warns(iris.warnings.IrisUserWarning, match=exc_regexp):
            _ = concatenate(sample_cubes, False)


class TestOrder:
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
        assert len(result) == 1

    def test_asc_bounds(self):
        top = self._make_cube([22.5, 67.5], [[0, 45], [45, 90]])
        bottom = self._make_cube([-67.5, -22.5], [[-90, -45], [-45, 0]])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_asc_points_with_singleton_ordered(self):
        top = self._make_cube([5])
        bottom = self._make_cube([15, 25])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_asc_points_with_singleton_unordered(self):
        top = self._make_cube([25])
        bottom = self._make_cube([5, 15])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_asc_bounds_with_singleton_ordered(self):
        top = self._make_cube([5], [[0, 10]])
        bottom = self._make_cube([15, 25], [[10, 20], [20, 30]])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_asc_bounds_with_singleton_unordered(self):
        top = self._make_cube([25], [[20, 30]])
        bottom = self._make_cube([5, 15], [[0, 10], [10, 20]])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_desc_points(self):
        top = self._make_cube([90, 70, 50, 30, 10])
        bottom = self._make_cube([-10, -30, -50, -70, -90])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_desc_bounds(self):
        top = self._make_cube([67.5, 22.5], [[90, 45], [45, 0]])
        bottom = self._make_cube([-22.5, -67.5], [[0, -45], [-45, -90]])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_desc_points_with_singleton_ordered(self):
        top = self._make_cube([25])
        bottom = self._make_cube([15, 5])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_desc_points_with_singleton_unordered(self):
        top = self._make_cube([5])
        bottom = self._make_cube([25, 15])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_desc_bounds_with_singleton_ordered(self):
        top = self._make_cube([25], [[30, 20]])
        bottom = self._make_cube([15, 5], [[20, 10], [10, 0]])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_desc_bounds_with_singleton_unordered(self):
        top = self._make_cube([5], [[10, 0]])
        bottom = self._make_cube([25, 15], [[30, 20], [20, 10]])
        result = concatenate([top, bottom])
        assert len(result) == 1

    def test_points_all_singleton(self):
        top = self._make_cube([5])
        bottom = self._make_cube([15])
        result1 = concatenate([top, bottom])
        result2 = concatenate([bottom, top])
        assert len(result1) == 1
        assert result1 == result2

    def test_asc_bounds_all_singleton(self):
        top = self._make_cube([5], [0, 10])
        bottom = self._make_cube([15], [10, 20])
        result1 = concatenate([top, bottom])
        result2 = concatenate([bottom, top])
        assert len(result1) == 1
        assert result1 == result2

    def test_desc_bounds_all_singleton(self):
        top = self._make_cube([5], [10, 0])
        bottom = self._make_cube([15], [20, 10])
        result1 = concatenate([top, bottom])
        result2 = concatenate([bottom, top])
        assert len(result1) == 1
        assert result1 == result2


class TestConcatenate__dask:
    @pytest.fixture()
    def sample_lazy_cubes(self):
        # Make a pair of concatenatable cubes, with dim points [1, 2] and [3, 4, 5]
        def build_lazy_cube(points):
            nx = 4
            data = np.arange(len(points) * nx).reshape(len(points), nx)
            data = as_lazy_data(data)
            cube = iris.cube.Cube(data, standard_name="air_temperature", units="K")
            lat = iris.coords.DimCoord(points, "latitude")
            lon = iris.coords.DimCoord(np.arange(nx), "longitude")
            cube.add_dim_coord(lat, 0)
            cube.add_dim_coord(lon, 1)
            return cube

        c1 = build_lazy_cube([1, 2])
        c2 = build_lazy_cube([3, 4, 5])
        return c1, c2

    @staticmethod
    def add_sample_auxcoord(cube):
        # Augment a test cube by adding an aux-coord on the concatenation dimension
        n_points, nx = cube.shape
        bounds = np.arange(n_points * nx * 4).reshape(n_points, nx, 4)
        bounds = as_lazy_data(bounds)
        aux_coord = iris.coords.AuxCoord(
            cube.core_data(),
            bounds=bounds,
            var_name="aux_coord",
        )
        cube.add_aux_coord(aux_coord, (0, 1))

    def test_lazy_concatenate(self, sample_lazy_cubes):
        (cube,) = concatenate(sample_lazy_cubes)
        assert cube.has_lazy_data()
        assert not ma.isMaskedArray(cube.data)

    def test_lazy_concatenate_aux_coords(self, sample_lazy_cubes):
        c1, c2 = sample_lazy_cubes
        for cube in (c1, c2):
            self.add_sample_auxcoord(cube)
        (result,) = concatenate([c1, c2])

        assert c1.coord("aux_coord").has_lazy_points()
        assert c1.coord("aux_coord").has_lazy_bounds()

        assert c2.coord("aux_coord").has_lazy_points()
        assert c2.coord("aux_coord").has_lazy_bounds()

        assert result.coord("aux_coord").has_lazy_points()
        assert result.coord("aux_coord").has_lazy_bounds()

    def test_lazy_concatenate_masked_array_mixed_deferred(self, sample_lazy_cubes):
        c1, c2 = sample_lazy_cubes
        c2.data = np.ma.masked_greater(c2.data, 3)
        (cube,) = concatenate([c1, c2])
        assert cube.has_lazy_data()
        assert ma.isMaskedArray(cube.data)
