# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for cube html representation."""

from html import escape

import numpy as np
import pytest

from iris.cube import Cube
from iris.experimental.representation import CubeRepresentation
from iris.tests import _shared_utils
import iris.tests.stock as stock


@_shared_utils.skip_data
class TestNoMetadata:
    # Test the situation where we have a cube with no metadata at all.
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.shape = (2, 3, 4)
        self.cube = Cube(np.arange(24).reshape(self.shape))
        self.representer = CubeRepresentation(self.cube)
        self.representer.repr_html()

    def test_cube_name(self):
        expected = "Unknown"  # This cube has no metadata.
        result = self.representer.name
        assert expected == result

    def test_cube_units(self):
        expected = "unknown"  # This cube has no metadata.
        result = self.representer.units
        assert expected == result

    def test_dim_names(self):
        expected = ["--"] * len(self.shape)
        result = self.representer.names
        assert expected == result

    def test_shape(self):
        result = self.representer.shapes
        assert result == self.shape


@_shared_utils.skip_data
class TestMissingMetadata:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = stock.realistic_3d()

    def test_no_coords(self):
        all_coords = [coord.name() for coord in self.cube.coords()]
        for coord in all_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        assert "dimension coordinates" not in result
        assert "auxiliary coordinates" not in result
        assert "scalar coordinates" not in result
        assert "attributes" in result

    def test_no_dim_coords(self):
        dim_coords = [c.name() for c in self.cube.coords(dim_coords=True)]
        for coord in dim_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        assert "dimension coordinates" not in result
        assert "auxiliary coordinates" in result
        assert "scalar coordinates" in result
        assert "attributes" in result

    def test_no_aux_coords(self):
        aux_coords = ["forecast_period"]
        for coord in aux_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        assert "dimension coordinates" in result
        assert "auxiliary coordinates" not in result
        assert "scalar coordinates" in result
        assert "attributes" in result

    def test_no_scalar_coords(self):
        aux_coords = ["air_pressure"]
        for coord in aux_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        assert "dimension coordinates" in result
        assert "auxiliary coordinates" in result
        assert "scalar coordinates" not in result
        assert "attributes" in result

    def test_no_attrs(self):
        self.cube.attributes = {}
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        assert "dimension coordinates" in result
        assert "auxiliary coordinates" in result
        assert "scalar coordinates" in result
        assert "attributes" not in result

    def test_no_cell_methods(self):
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        assert "cell methods" not in result


@_shared_utils.skip_data
class TestScalarCube:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = stock.realistic_3d()[0, 0, 0]
        self.representer = CubeRepresentation(self.cube)
        self.representer.repr_html()

    def test_identfication(self):
        # Is this scalar cube accurately identified?
        assert self.representer.scalar_cube

    def test_header__name(self):
        header = self.representer._make_header()
        expected_name = escape(self.cube.name().title().replace("_", " "))
        assert expected_name in header

    def test_header__units(self):
        header = self.representer._make_header()
        expected_units = escape(self.cube.units.symbol)
        assert expected_units in header

    def test_header__scalar_str(self):
        # Check that 'scalar cube' is placed in the header.
        header = self.representer._make_header()
        expected_str = "(scalar cube)"
        assert expected_str in header

    def test_content__scalars(self):
        # Check an element "Scalar coordinates" is present in the main content.
        content = self.representer._make_content()
        expected_str = "Scalar coordinates"
        assert expected_str in content

    def test_content__specific_scalar_coord(self):
        # Check a specific scalar coord is present in the main content.
        content = self.representer._make_content()
        expected_coord = self.cube.coords()[0]
        expected_coord_name = escape(expected_coord.name())
        assert expected_coord_name in content
        expected_coord_val = escape(str(expected_coord.points[0]))
        assert expected_coord_val in content

    def test_content__attributes(self):
        # Check an element "attributes" is present in the main content.
        content = self.representer._make_content()
        expected_str = "Attributes"
        assert expected_str in content
