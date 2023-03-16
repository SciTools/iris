# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.new_axis`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
# isort: off
import iris.tests as tests  # noqa

# isort: on
import copy

import numpy as np
import pytest

import iris
from iris._lazy_data import as_lazy_data
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import Cube
import iris.tests.stock as stock
from iris.util import new_axis


class Test:
    @pytest.fixture
    def stock_cube(self):
        cube = stock.simple_2d_w_cell_measure_ancil_var()
        time = iris.coords.DimCoord([1], standard_name="time")
        cube.add_aux_coord(time, None)
        cube.coord("wibble").bounds = np.array([0, 2]).reshape((1, 2))
        return cube

    def _assert_cube_notis(self, cube_a, cube_b):
        assert cube_a.metadata is not cube_b.metadata

        for coord_a, coord_b in zip(cube_a.coords(), cube_b.coords()):
            assert coord_a is not coord_b

        for av_a, av_b in zip(
            cube_a.ancillary_variables(), cube_b.ancillary_variables()
        ):
            assert av_a is not av_b

        for cm_a, cm_b in zip(cube_a.cell_measures(), cube_b.cell_measures()):
            assert cm_a is not cm_b

        for factory_a, factory_b in zip(
            cube_a.aux_factories, cube_b.aux_factories
        ):
            assert factory_a is not factory_b

    def test_promote_no_coord(self, stock_cube):
        # Providing no coordinate to promote.
        result = new_axis(stock_cube)
        expected = iris.cube.Cube(
            stock_cube.data[None], long_name="thingness", units="1"
        )
        expected.add_dim_coord(stock_cube.coord("bar").copy(), 1)
        expected.add_dim_coord(stock_cube.coord("foo").copy(), 2)
        expected.add_aux_coord(stock_cube.coord("time").copy(), None)
        expected.add_aux_coord(stock_cube.coord("wibble").copy(), None)
        expected.add_ancillary_variable(
            stock_cube.ancillary_variable("quality_flag"), 1
        )
        expected.add_cell_measure(stock_cube.cell_measure("cell_area"), (1, 2))

        assert result == expected
        self._assert_cube_notis(result, stock_cube)

    def test_promote_scalar_dimcoord(self, stock_cube):
        # Providing a scalar coordinate to promote.
        result = new_axis(stock_cube, "time")
        expected = iris.cube.Cube(
            stock_cube.data[None], long_name="thingness", units="1"
        )
        expected.add_dim_coord(stock_cube.coord("bar").copy(), 1)
        expected.add_dim_coord(stock_cube.coord("foo").copy(), 2)
        expected.add_aux_coord(stock_cube.coord("time").copy(), 0)
        expected.add_aux_coord(stock_cube.coord("wibble").copy(), None)
        expected.add_ancillary_variable(
            stock_cube.ancillary_variable("quality_flag"), 1
        )
        expected.add_cell_measure(stock_cube.cell_measure("cell_area"), (1, 2))

        assert result == expected
        # Explicitly check time has been made a cube dim coord as cube equality
        # does not check this.
        assert result.coord("time") in [
            item[0] for item in result._dim_coords_and_dims
        ]
        self._assert_cube_notis(result, stock_cube)

    def test_promote_scalar_auxcoord(self, stock_cube):
        # Providing a scalar coordinate to promote.
        result = new_axis(stock_cube, "wibble")
        expected = iris.cube.Cube(
            stock_cube.data[None], long_name="thingness", units="1"
        )
        expected.add_dim_coord(stock_cube.coord("bar").copy(), 1)
        expected.add_dim_coord(stock_cube.coord("foo").copy(), 2)
        expected.add_aux_coord(stock_cube.coord("time").copy(), None)
        expected.add_aux_coord(stock_cube.coord("wibble").copy(), 0)
        expected.add_ancillary_variable(
            stock_cube.ancillary_variable("quality_flag"), 1
        )
        expected.add_cell_measure(stock_cube.cell_measure("cell_area"), (1, 2))

        assert result == expected
        # Explicitly check wibble has been made a cube dim coord as cube
        # equality does not check this.
        assert result.coord("wibble") in [
            item[0] for item in result._dim_coords_and_dims
        ]
        self._assert_cube_notis(result, stock_cube)

    def test_existing_dim_coord(self, stock_cube):
        # Provide an existing dimensional coordinate
        coord = iris.coords.DimCoord(1, long_name="dim")
        stock_cube.add_aux_coord(coord)

        new_cube = iris.util.new_axis(stock_cube, coord)
        with pytest.raises(
            ValueError, match="is already a dimension coordinate."
        ):
            iris.util.new_axis(new_cube, coord)

    def test_promote_non_scalar(self, stock_cube):
        # Provide a dimensional coordinate which is not scalar
        iris.util.demote_dim_coord_to_aux_coord(stock_cube, "foo")
        with pytest.raises(ValueError, match="is not a scalar coordinate."):
            new_axis(stock_cube, "foo")

    def test_maint_factory(self):
        # Ensure that aux factory persists.
        data = np.arange(12, dtype="i8").reshape((3, 4))

        orography = AuxCoord(
            [10, 25, 50, 5], standard_name="surface_altitude", units="m"
        )

        model_level = AuxCoord([2, 1, 0], standard_name="model_level_number")

        level_height = DimCoord(
            [100, 50, 10],
            long_name="level_height",
            units="m",
            attributes={"positive": "up"},
            bounds=[[150, 75], [75, 20], [20, 0]],
        )

        sigma = AuxCoord(
            [0.8, 0.9, 0.95],
            long_name="sigma",
            bounds=[[0.7, 0.85], [0.85, 0.97], [0.97, 1.0]],
        )

        hybrid_height = iris.aux_factory.HybridHeightFactory(
            level_height, sigma, orography
        )

        cube = Cube(
            data,
            standard_name="air_temperature",
            units="K",
            dim_coords_and_dims=[(level_height, 0)],
            aux_coords_and_dims=[(orography, 1), (model_level, 0), (sigma, 0)],
            aux_factories=[hybrid_height],
        )

        com = Cube(
            data[None],
            standard_name="air_temperature",
            units="K",
            dim_coords_and_dims=[(copy.copy(level_height), 1)],
            aux_coords_and_dims=[
                (copy.copy(orography), 2),
                (copy.copy(model_level), 1),
                (copy.copy(sigma), 1),
            ],
            aux_factories=[copy.copy(hybrid_height)],
        )
        res = new_axis(cube)

        assert res == com
        self._assert_cube_notis(res, cube)

        # Check that factory dependencies are actual coords within the cube.
        # Addresses a former bug : https://github.com/SciTools/iris/pull/3263
        (factory,) = list(res.aux_factories)
        deps = factory.dependencies
        for dep_name, dep_coord in deps.items():
            coord_name = dep_coord.name()
            assert dep_coord is res.coord(coord_name)

    def test_lazy_cube_data(self, stock_cube):
        stock_cube.data = as_lazy_data(stock_cube.data)
        res = new_axis(stock_cube)
        assert stock_cube.has_lazy_data()
        assert res.has_lazy_data()
        assert res.shape == (1,) + stock_cube.shape

    def test_masked_unit_array(self):
        cube = stock.simple_3d_mask()
        test_cube = cube[0, 0, 0]
        test_cube = new_axis(test_cube, "longitude")
        test_cube = new_axis(test_cube, "latitude")
        data_shape = test_cube.data.shape
        mask_shape = test_cube.data.mask.shape
        assert data_shape == mask_shape

    def test_expand_scalar_coord(self, stock_cube):
        result = new_axis(stock_cube, "time", expand_extras=["wibble"])

        expected = iris.cube.Cube(
            stock_cube.data[None], long_name="thingness", units="1"
        )
        expected.add_dim_coord(stock_cube.coord("bar").copy(), 1)
        expected.add_dim_coord(stock_cube.coord("foo").copy(), 2)
        expected.add_aux_coord(stock_cube.coord("time").copy(), 0)
        expected.add_aux_coord(stock_cube.coord("wibble").copy(), 0)
        expected.add_ancillary_variable(
            stock_cube.ancillary_variable("quality_flag"), 1
        )
        expected.add_cell_measure(stock_cube.cell_measure("cell_area"), (1, 2))

        assert result == expected
        self._assert_cube_notis(result, stock_cube)

    def test_expand_scalar_coord_lazy_points(self, stock_cube):
        stock_cube.coord("wibble").points = as_lazy_data(
            stock_cube.coord("wibble").points
        )
        result = new_axis(stock_cube, "time", expand_extras=["wibble"])
        assert stock_cube.coord("wibble").has_lazy_points()
        assert result.coord("wibble").has_lazy_points()
        assert (
            result.coord("wibble").points.shape
            == stock_cube.coord("wibble").points.shape
        )

    def test_expand_scalar_coord_lazy_bounds(self, stock_cube):
        stock_cube.coord("wibble").bounds = as_lazy_data(np.array([[0, 2]]))
        result = new_axis(stock_cube, "time", expand_extras=["wibble"])
        assert stock_cube.coord("wibble").has_lazy_bounds()
        assert result.coord("wibble").has_lazy_bounds()
        assert (
            result.coord("wibble").bounds.shape
            == stock_cube.coord("wibble").bounds.shape
        )

    def test_expand_cell_measure(self, stock_cube):
        result = new_axis(stock_cube, "time", expand_extras=["cell_area"])

        expected = iris.cube.Cube(
            stock_cube.data[None], long_name="thingness", units="1"
        )
        expected.add_dim_coord(stock_cube.coord("bar").copy(), 1)
        expected.add_dim_coord(stock_cube.coord("foo").copy(), 2)
        expected.add_aux_coord(stock_cube.coord("time").copy(), 0)
        expected.add_aux_coord(stock_cube.coord("wibble").copy(), None)
        expected.add_ancillary_variable(
            stock_cube.ancillary_variable("quality_flag"), 1
        )

        expected_cm = CellMeasure(
            stock_cube.cell_measure("cell_area").data[None],
            standard_name="cell_area",
        )
        expected.add_cell_measure(expected_cm, (0, 1, 2))

        assert result == expected
        self._assert_cube_notis(result, stock_cube)

    def test_expand_ancil_var(self, stock_cube):
        result = new_axis(stock_cube, "time", expand_extras=["quality_flag"])

        expected = iris.cube.Cube(
            stock_cube.data[None], long_name="thingness", units="1"
        )
        expected.add_dim_coord(stock_cube.coord("bar").copy(), 1)
        expected.add_dim_coord(stock_cube.coord("foo").copy(), 2)
        expected.add_aux_coord(stock_cube.coord("time").copy(), 0)
        expected.add_aux_coord(stock_cube.coord("wibble").copy(), None)
        expected.add_cell_measure(stock_cube.cell_measure("cell_area"), (1, 2))

        expected_av = AncillaryVariable(
            stock_cube.ancillary_variable("quality_flag").data[None],
            standard_name="quality_flag",
        )

        expected.add_ancillary_variable(expected_av, (0, 1))

        assert result == expected
        self._assert_cube_notis(result, stock_cube)

    def test_expand_multiple(self, stock_cube):
        result = new_axis(
            stock_cube, "time", expand_extras=["wibble", "cell_area"]
        )

        expected = iris.cube.Cube(
            stock_cube.data[None], long_name="thingness", units="1"
        )
        expected.add_dim_coord(stock_cube.coord("bar").copy(), 1)
        expected.add_dim_coord(stock_cube.coord("foo").copy(), 2)
        expected.add_aux_coord(stock_cube.coord("time").copy(), 0)
        expected.add_aux_coord(stock_cube.coord("wibble").copy(), 0)
        expected.add_ancillary_variable(
            stock_cube.ancillary_variable("quality_flag"), 1
        )

        expected_cm = CellMeasure(
            stock_cube.cell_measure("cell_area").data[None],
            standard_name="cell_area",
        )
        expected.add_cell_measure(expected_cm, (0, 1, 2))

        assert result == expected
        self._assert_cube_notis(result, stock_cube)
