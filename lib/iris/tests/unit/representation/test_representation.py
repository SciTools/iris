# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris._representation` module."""

import numpy as np
import iris.tests as tests
import iris._representation
from iris.cube import Cube
from iris.coords import (
    DimCoord,
    AuxCoord,
    CellMeasure,
    AncillaryVariable,
    CellMethod,
)


def example_cube():
    cube = Cube(
        np.arange(6).reshape([3, 2]),
        standard_name="air_temperature",
        long_name="screen_air_temp",
        var_name="airtemp",
        units="K",
    )
    lat = DimCoord([0, 1, 2], standard_name="latitude", units="degrees")
    cube.add_dim_coord(lat, 0)
    return cube


class Test_CubeSummary(tests.IrisTest):
    def setUp(self):
        self.cube = example_cube()

    def test_header(self):
        rep = iris._representation.CubeSummary(self.cube)
        header_left = rep.header.nameunit
        header_right = rep.header.dimension_header.contents

        self.assertEqual(header_left, "air_temperature / (K)")
        self.assertEqual(header_right, ["latitude: 3", "-- : 2"])

    def test_blank_cube(self):
        cube = Cube([1, 2])
        rep = iris._representation.CubeSummary(cube)

        self.assertEqual(rep.header.nameunit, "unknown / (unknown)")
        self.assertEqual(rep.header.dimension_header.contents, ["-- : 2"])

        expected_vector_sections = [
            "Dimension coordinates:",
            "Auxiliary coordinates:",
            "Derived coordinates:",
            "Cell Measures:",
            "Ancillary Variables:",
        ]
        self.assertEqual(
            list(rep.vector_sections.keys()), expected_vector_sections
        )
        for title in expected_vector_sections:
            vector_section = rep.vector_sections[title]
            self.assertEqual(vector_section.contents, [])
            self.assertTrue(vector_section.is_empty())

        expected_scalar_sections = [
            "Scalar Coordinates:",
            "Scalar cell measures:",
            "Attributes:",
            "Cell methods:",
        ]

        self.assertEqual(
            list(rep.scalar_sections.keys()), expected_scalar_sections
        )
        for title in expected_scalar_sections:
            scalar_section = rep.scalar_sections[title]
            self.assertEqual(scalar_section.contents, [])
            self.assertTrue(scalar_section.is_empty())

    def test_vector_coord(self):
        rep = iris._representation.CubeSummary(self.cube)
        dim_section = rep.vector_sections["Dimension coordinates:"]

        self.assertEqual(len(dim_section.contents), 1)
        self.assertFalse(dim_section.is_empty())

        dim_summary = dim_section.contents[0]

        name = dim_summary.name
        dim_chars = dim_summary.dim_chars
        extra = dim_summary.extra

        self.assertEqual(name, "latitude")
        self.assertEqual(dim_chars, ["x", "-"])
        self.assertEqual(extra, "")

    def test_scalar_coord(self):
        cube = self.cube
        scalar_coord_no_bounds = AuxCoord([10], long_name="bar", units="K")
        scalar_coord_with_bounds = AuxCoord(
            [10], long_name="foo", units="K", bounds=[(5, 15)]
        )
        scalar_coord_text = AuxCoord(
            ["a\nb\nc"], long_name="foo", attributes={"key": "value"}
        )
        cube.add_aux_coord(scalar_coord_no_bounds)
        cube.add_aux_coord(scalar_coord_with_bounds)
        cube.add_aux_coord(scalar_coord_text)
        rep = iris._representation.CubeSummary(cube)

        scalar_section = rep.scalar_sections["Scalar Coordinates:"]

        self.assertEqual(len(scalar_section.contents), 3)

        no_bounds_summary = scalar_section.contents[0]
        bounds_summary = scalar_section.contents[1]
        text_summary = scalar_section.contents[2]

        self.assertEqual(no_bounds_summary.name, "bar")
        self.assertEqual(no_bounds_summary.content, "10 K")
        self.assertEqual(no_bounds_summary.extra, "")

        self.assertEqual(bounds_summary.name, "foo")
        self.assertEqual(bounds_summary.content, "10 K, bound=(5, 15) K")
        self.assertEqual(bounds_summary.extra, "")

        self.assertEqual(text_summary.name, "foo")
        self.assertEqual(text_summary.content, "a\nb\nc")
        self.assertEqual(text_summary.extra, "key='value'")

    def test_cell_measure(self):
        cube = self.cube
        cell_measure = CellMeasure([1, 2, 3], long_name="foo")
        cube.add_cell_measure(cell_measure, 0)
        rep = iris._representation.CubeSummary(cube)

        cm_section = rep.vector_sections["Cell Measures:"]
        self.assertEqual(len(cm_section.contents), 1)

        cm_summary = cm_section.contents[0]
        self.assertEqual(cm_summary.name, "foo")
        self.assertEqual(cm_summary.dim_chars, ["x", "-"])

    def test_ancillary_variable(self):
        cube = self.cube
        cell_measure = AncillaryVariable([1, 2, 3], long_name="foo")
        cube.add_ancillary_variable(cell_measure, 0)
        rep = iris._representation.CubeSummary(cube)

        av_section = rep.vector_sections["Ancillary Variables:"]
        self.assertEqual(len(av_section.contents), 1)

        av_summary = av_section.contents[0]
        self.assertEqual(av_summary.name, "foo")
        self.assertEqual(av_summary.dim_chars, ["x", "-"])

    def test_attributes(self):
        cube = self.cube
        cube.attributes = {"a": 1, "b": "two"}
        rep = iris._representation.CubeSummary(cube)

        attribute_section = rep.scalar_sections["Attributes:"]
        attribute_contents = attribute_section.contents
        expected_contents = ["a: 1", "b: two"]

        self.assertEqual(attribute_contents, expected_contents)

    def test_cell_methods(self):
        cube = self.cube
        x = AuxCoord(1, long_name="x")
        y = AuxCoord(1, long_name="y")
        cell_method_xy = CellMethod("mean", [x, y])
        cell_method_x = CellMethod("mean", x)
        cube.add_cell_method(cell_method_xy)
        cube.add_cell_method(cell_method_x)

        rep = iris._representation.CubeSummary(cube)
        cell_method_section = rep.scalar_sections["Cell methods:"]
        expected_contents = ["mean: x, y", "mean: x"]
        self.assertEqual(cell_method_section.contents, expected_contents)


if __name__ == "__main__":
    tests.main()
