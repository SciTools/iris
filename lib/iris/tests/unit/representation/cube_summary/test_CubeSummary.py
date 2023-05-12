# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris._representation.cube_summary.CubeSummary`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris._representation.cube_summary import CubeSummary
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    CellMethod,
    DimCoord,
)
from iris.cube import Cube
from iris.tests.stock.mesh import sample_mesh_cube


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
        rep = CubeSummary(self.cube)
        header_left = rep.header.nameunit
        header_right = rep.header.dimension_header.contents

        self.assertEqual(header_left, "air_temperature / (K)")
        self.assertEqual(header_right, ["latitude: 3", "-- : 2"])

    def test_blank_cube(self):
        cube = Cube([1, 2])
        rep = CubeSummary(cube)

        self.assertEqual(rep.header.nameunit, "unknown / (unknown)")
        self.assertEqual(rep.header.dimension_header.contents, ["-- : 2"])

        expected_vector_sections = [
            "Dimension coordinates:",
            "Mesh coordinates:",
            "Auxiliary coordinates:",
            "Derived coordinates:",
            "Cell measures:",
            "Ancillary variables:",
        ]
        self.assertEqual(
            list(rep.vector_sections.keys()), expected_vector_sections
        )
        for title in expected_vector_sections:
            vector_section = rep.vector_sections[title]
            self.assertEqual(vector_section.contents, [])
            self.assertTrue(vector_section.is_empty())

        expected_scalar_sections = [
            "Mesh:",
            "Scalar coordinates:",
            "Scalar cell measures:",
            "Scalar ancillary variables:",
            "Cell methods:",
            "Attributes:",
        ]

        self.assertEqual(
            list(rep.scalar_sections.keys()), expected_scalar_sections
        )
        for title in expected_scalar_sections:
            scalar_section = rep.scalar_sections[title]
            self.assertEqual(scalar_section.contents, [])
            self.assertTrue(scalar_section.is_empty())

    def test_vector_coord(self):
        rep = CubeSummary(self.cube)
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
        scalar_coord_simple_text = AuxCoord(
            ["this and that"],
            long_name="foo",
            attributes={"key": 42, "key2": "value-str"},
        )
        scalar_coord_awkward_text = AuxCoord(
            ["a is\nb\n and c"], long_name="foo_2"
        )
        cube.add_aux_coord(scalar_coord_no_bounds)
        cube.add_aux_coord(scalar_coord_with_bounds)
        cube.add_aux_coord(scalar_coord_simple_text)
        cube.add_aux_coord(scalar_coord_awkward_text)
        rep = CubeSummary(cube)

        scalar_section = rep.scalar_sections["Scalar coordinates:"]

        self.assertEqual(len(scalar_section.contents), 4)

        no_bounds_summary = scalar_section.contents[0]
        bounds_summary = scalar_section.contents[1]
        text_summary_simple = scalar_section.contents[2]
        text_summary_awkward = scalar_section.contents[3]

        self.assertEqual(no_bounds_summary.name, "bar")
        self.assertEqual(no_bounds_summary.content, "10 K")
        self.assertEqual(no_bounds_summary.extra, "")

        self.assertEqual(bounds_summary.name, "foo")
        self.assertEqual(bounds_summary.content, "10 K, bound=(5, 15) K")
        self.assertEqual(bounds_summary.extra, "")

        self.assertEqual(text_summary_simple.name, "foo")
        self.assertEqual(text_summary_simple.content, "this and that")
        self.assertEqual(text_summary_simple.lines, ["this and that"])
        self.assertEqual(text_summary_simple.extra, "key=42, key2='value-str'")

        self.assertEqual(text_summary_awkward.name, "foo_2")
        self.assertEqual(text_summary_awkward.content, r"'a is\nb\n and c'")
        self.assertEqual(text_summary_awkward.lines, ["a is", "b", " and c"])
        self.assertEqual(text_summary_awkward.extra, "")

    def test_cell_measure(self):
        cube = self.cube
        cell_measure = CellMeasure([1, 2, 3], long_name="foo")
        cube.add_cell_measure(cell_measure, 0)
        rep = CubeSummary(cube)

        cm_section = rep.vector_sections["Cell measures:"]
        self.assertEqual(len(cm_section.contents), 1)

        cm_summary = cm_section.contents[0]
        self.assertEqual(cm_summary.name, "foo")
        self.assertEqual(cm_summary.dim_chars, ["x", "-"])

    def test_ancillary_variable(self):
        cube = self.cube
        cell_measure = AncillaryVariable([1, 2, 3], long_name="foo")
        cube.add_ancillary_variable(cell_measure, 0)
        rep = CubeSummary(cube)

        av_section = rep.vector_sections["Ancillary variables:"]
        self.assertEqual(len(av_section.contents), 1)

        av_summary = av_section.contents[0]
        self.assertEqual(av_summary.name, "foo")
        self.assertEqual(av_summary.dim_chars, ["x", "-"])

    def test_attributes(self):
        cube = self.cube
        cube.attributes = {"a": 1, "b": "two", "c": " this \n   that\tand."}
        rep = CubeSummary(cube)

        attribute_section = rep.scalar_sections["Attributes:"]
        attribute_contents = attribute_section.contents
        expected_contents = [
            "a: 1",
            "b: 'two'",
            "c: ' this \\n   that\\tand.'",
        ]
        # Note: a string with \n or \t in it gets "repr-d".
        # Other strings don't (though in coord 'extra' lines, they do.)

        self.assertEqual(attribute_contents, expected_contents)

    def test_cell_methods(self):
        cube = self.cube
        x = AuxCoord(1, long_name="x")
        y = AuxCoord(1, long_name="y")
        cell_method_xy = CellMethod("mean", [x, y])
        cell_method_x = CellMethod("mean", x)
        cube.add_cell_method(cell_method_xy)
        cube.add_cell_method(cell_method_x)

        rep = CubeSummary(cube)
        cell_method_section = rep.scalar_sections["Cell methods:"]
        expected_contents = ["0: x: y: mean", "1: x: mean"]
        self.assertEqual(cell_method_section.contents, expected_contents)

    def test_scalar_cube(self):
        cube = self.cube
        while cube.ndim > 0:
            cube = cube[0]
        rep = CubeSummary(cube)
        self.assertEqual(rep.header.nameunit, "air_temperature / (K)")
        self.assertTrue(rep.header.dimension_header.scalar)
        self.assertEqual(rep.header.dimension_header.dim_names, [])
        self.assertEqual(rep.header.dimension_header.shape, [])
        self.assertEqual(rep.header.dimension_header.contents, ["scalar cube"])
        self.assertEqual(len(rep.vector_sections), 6)
        self.assertTrue(
            all(sect.is_empty() for sect in rep.vector_sections.values())
        )
        self.assertEqual(len(rep.scalar_sections), 6)
        self.assertEqual(
            len(rep.scalar_sections["Scalar coordinates:"].contents), 1
        )
        self.assertTrue(
            rep.scalar_sections["Scalar cell measures:"].is_empty()
        )
        self.assertTrue(rep.scalar_sections["Attributes:"].is_empty())
        self.assertTrue(rep.scalar_sections["Cell methods:"].is_empty())

    def test_coord_attributes(self):
        cube = self.cube
        co1 = cube.coord("latitude")
        co1.attributes.update(dict(a=1, b=2))
        co2 = co1.copy()
        co2.attributes.update(dict(a=7, z=77, text="ok", text2="multi\nline"))
        cube.add_aux_coord(co2, cube.coord_dims(co1))
        rep = CubeSummary(cube)
        co1_summ = rep.vector_sections["Dimension coordinates:"].contents[0]
        co2_summ = rep.vector_sections["Auxiliary coordinates:"].contents[0]
        # Notes: 'b' is same so does not appear; sorted order; quoted strings.
        self.assertEqual(co1_summ.extra, "a=1")
        self.assertEqual(
            co2_summ.extra, "a=7, text='ok', text2='multi\\nline', z=77"
        )

    def test_array_attributes(self):
        cube = self.cube
        co1 = cube.coord("latitude")
        co1.attributes.update(dict(a=1, array=np.array([1.2, 3])))
        co2 = co1.copy()
        co2.attributes.update(dict(b=2, array=np.array([3.2, 1])))
        cube.add_aux_coord(co2, cube.coord_dims(co1))
        rep = CubeSummary(cube)
        co1_summ = rep.vector_sections["Dimension coordinates:"].contents[0]
        co2_summ = rep.vector_sections["Auxiliary coordinates:"].contents[0]
        self.assertEqual(co1_summ.extra, "array=array([1.2, 3. ])")
        self.assertEqual(co2_summ.extra, "array=array([3.2, 1. ]), b=2")

    def test_attributes_subtle_differences(self):
        cube = Cube([0])

        # Add a pair that differ only in having a list instead of an array.
        co1a = DimCoord(
            [0],
            long_name="co1_list_or_array",
            attributes=dict(x=1, arr1=np.array(2), arr2=np.array([1, 2])),
        )
        co1b = co1a.copy()
        co1b.attributes.update(dict(arr2=[1, 2]))
        for co in (co1a, co1b):
            cube.add_aux_coord(co)

        # Add a pair that differ only in an attribute array dtype.
        co2a = AuxCoord(
            [0],
            long_name="co2_dtype",
            attributes=dict(x=1, arr1=np.array(2), arr2=np.array([3, 4])),
        )
        co2b = co2a.copy()
        co2b.attributes.update(dict(arr2=np.array([3.0, 4.0])))
        assert co2b != co2a
        for co in (co2a, co2b):
            cube.add_aux_coord(co)

        # Add a pair that differ only in an attribute array shape.
        co3a = DimCoord(
            [0],
            long_name="co3_shape",
            attributes=dict(x=1, arr1=np.array([5, 6]), arr2=np.array([3, 4])),
        )
        co3b = co3a.copy()
        co3b.attributes.update(dict(arr1=np.array([[5], [6]])))
        for co in (co3a, co3b):
            cube.add_aux_coord(co)

        rep = CubeSummary(cube)
        co_summs = rep.scalar_sections["Scalar coordinates:"].contents
        co1a_summ, co1b_summ = co_summs[0:2]
        self.assertEqual(co1a_summ.extra, "arr2=array([1, 2])")
        self.assertEqual(co1b_summ.extra, "arr2=[1, 2]")
        co2a_summ, co2b_summ = co_summs[2:4]
        self.assertEqual(co2a_summ.extra, "arr2=array([3, 4])")
        self.assertEqual(co2b_summ.extra, "arr2=array([3., 4.])")
        co3a_summ, co3b_summ = co_summs[4:6]
        self.assertEqual(co3a_summ.extra, "arr1=array([5, 6])")
        self.assertEqual(co3b_summ.extra, "arr1=array([[5], [6]])")

    def test_unstructured_cube(self):
        cube = sample_mesh_cube()
        rep = CubeSummary(cube)
        # Just check that coordinates appear in the expected sections
        dim_section = rep.vector_sections["Dimension coordinates:"]
        mesh_section = rep.vector_sections["Mesh coordinates:"]
        aux_section = rep.vector_sections["Auxiliary coordinates:"]
        self.assertEqual(len(dim_section.contents), 2)
        self.assertEqual(len(mesh_section.contents), 2)
        self.assertEqual(len(aux_section.contents), 1)


if __name__ == "__main__":
    tests.main()
