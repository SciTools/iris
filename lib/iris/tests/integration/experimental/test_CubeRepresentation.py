# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for cube html representation."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from html import escape

import numpy as np

from iris.cube import Cube
from iris.experimental.representation import CubeRepresentation
import iris.tests.stock as stock


@tests.skip_data
class TestNoMetadata(tests.IrisTest):
    # Test the situation where we have a cube with no metadata at all.
    def setUp(self):
        self.shape = (2, 3, 4)
        self.cube = Cube(np.arange(24).reshape(self.shape))
        self.representer = CubeRepresentation(self.cube)
        self.representer.repr_html()

    def test_cube_name(self):
        expected = "Unknown"  # This cube has no metadata.
        result = self.representer.name
        self.assertEqual(expected, result)

    def test_cube_units(self):
        expected = "unknown"  # This cube has no metadata.
        result = self.representer.units
        self.assertEqual(expected, result)

    def test_dim_names(self):
        expected = ["--"] * len(self.shape)
        result = self.representer.names
        self.assertEqual(expected, result)

    def test_shape(self):
        result = self.representer.shapes
        self.assertEqual(result, self.shape)


@tests.skip_data
class TestMissingMetadata(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_3d()

    def test_no_coords(self):
        all_coords = [coord.name() for coord in self.cube.coords()]
        for coord in all_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertNotIn("dimension coordinates", result)
        self.assertNotIn("auxiliary coordinates", result)
        self.assertNotIn("scalar coordinates", result)
        self.assertIn("attributes", result)

    def test_no_dim_coords(self):
        dim_coords = [c.name() for c in self.cube.coords(dim_coords=True)]
        for coord in dim_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertNotIn("dimension coordinates", result)
        self.assertIn("auxiliary coordinates", result)
        self.assertIn("scalar coordinates", result)
        self.assertIn("attributes", result)

    def test_no_aux_coords(self):
        aux_coords = ["forecast_period"]
        for coord in aux_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertIn("dimension coordinates", result)
        self.assertNotIn("auxiliary coordinates", result)
        self.assertIn("scalar coordinates", result)
        self.assertIn("attributes", result)

    def test_no_scalar_coords(self):
        aux_coords = ["air_pressure"]
        for coord in aux_coords:
            self.cube.remove_coord(coord)
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertIn("dimension coordinates", result)
        self.assertIn("auxiliary coordinates", result)
        self.assertNotIn("scalar coordinates", result)
        self.assertIn("attributes", result)

    def test_no_attrs(self):
        self.cube.attributes = {}
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertIn("dimension coordinates", result)
        self.assertIn("auxiliary coordinates", result)
        self.assertIn("scalar coordinates", result)
        self.assertNotIn("attributes", result)

    def test_no_cell_methods(self):
        representer = CubeRepresentation(self.cube)
        result = representer.repr_html().lower()
        self.assertNotIn("cell methods", result)


@tests.skip_data
class TestScalarCube(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_3d()[0, 0, 0]
        self.representer = CubeRepresentation(self.cube)
        self.representer.repr_html()

    def test_identfication(self):
        # Is this scalar cube accurately identified?
        self.assertTrue(self.representer.scalar_cube)

    def test_header__name(self):
        header = self.representer._make_header()
        expected_name = escape(self.cube.name().title().replace("_", " "))
        self.assertIn(expected_name, header)

    def test_header__units(self):
        header = self.representer._make_header()
        expected_units = escape(self.cube.units.symbol)
        self.assertIn(expected_units, header)

    def test_header__scalar_str(self):
        # Check that 'scalar cube' is placed in the header.
        header = self.representer._make_header()
        expected_str = "(scalar cube)"
        self.assertIn(expected_str, header)

    def test_content__scalars(self):
        # Check an element "Scalar coordinates" is present in the main content.
        content = self.representer._make_content()
        expected_str = "Scalar coordinates"
        self.assertIn(expected_str, content)

    def test_content__specific_scalar_coord(self):
        # Check a specific scalar coord is present in the main content.
        content = self.representer._make_content()
        expected_coord = self.cube.coords()[0]
        expected_coord_name = escape(expected_coord.name())
        self.assertIn(expected_coord_name, content)
        expected_coord_val = escape(str(expected_coord.points[0]))
        self.assertIn(expected_coord_val, content)

    def test_content__attributes(self):
        # Check an element "attributes" is present in the main content.
        content = self.representer._make_content()
        expected_str = "Attributes"
        self.assertIn(expected_str, content)


if __name__ == "__main__":
    tests.main()
