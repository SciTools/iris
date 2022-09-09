# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris._representation.cube_printout.CubePrintout`."""
import iris.tests as tests  # isort:skip

import numpy as np

from iris._representation.cube_printout import CubePrinter
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


class TestCubePrintout___str__(tests.IrisTest):
    def test_str(self):
        # Just check that its str representation is the 'to_string' result.
        cube = Cube(0)
        printer = CubePrinter(CubeSummary(cube))
        result = str(printer)
        self.assertEqual(result, printer.to_string())


def cube_replines(cube, **kwargs):
    return CubePrinter(cube).to_string(**kwargs).split("\n")


class TestCubePrintout__to_string(tests.IrisTest):
    def test_empty(self):
        cube = Cube([0])
        rep = cube_replines(cube)
        expect = ["unknown / (unknown)                 (-- : 1)"]
        self.assertEqual(expect, rep)

    def test_shortform__default(self):
        cube = Cube([0])
        expect = ["unknown / (unknown)                 (-- : 1)"]
        # In this case, default one-line is the same.
        rep = cube_replines(cube, oneline=True)
        self.assertEqual(expect, rep)

    def test_shortform__compressed(self):
        cube = Cube([0])
        rep = cube_replines(cube, oneline=True, name_padding=0)
        expect = ["unknown / (unknown) (-- : 1)"]
        self.assertEqual(rep, expect)

    def _sample_wide_cube(self):
        cube = Cube([0, 1])
        cube.add_aux_coord(
            AuxCoord(
                [0, 1],
                long_name="long long long long long long long long name",
            ),
            0,
        )
        return cube

    def test_wide_cube(self):
        # For comparison with the shortform and padding-controlled cases.
        cube = self._sample_wide_cube()
        rep = cube_replines(cube)
        expect_full = [
            "unknown / (unknown)                                  (-- : 2)",
            "    Auxiliary coordinates:",
            "        long long long long long long long long name     x",
        ]
        self.assertEqual(expect_full, rep)

    def test_shortform__wide__default(self):
        cube = self._sample_wide_cube()
        rep = cube_replines(cube, oneline=True)
        # *default* one-line is shorter than full header, but not minimal.
        expect = ["unknown / (unknown)                 (-- : 2)"]
        self.assertEqual(rep, expect)

    def test_shortform__wide__compressed(self):
        cube = self._sample_wide_cube()
        rep = cube_replines(cube, oneline=True, name_padding=0)
        expect = ["unknown / (unknown) (-- : 2)"]
        self.assertEqual(rep, expect)

    def test_shortform__wide__intermediate(self):
        cube = self._sample_wide_cube()
        rep = cube_replines(cube, oneline=True, name_padding=25)
        expect = ["unknown / (unknown)       (-- : 2)"]
        self.assertEqual(expect, rep)

    def test_scalar_cube_summaries(self):
        cube = Cube(0)
        expect = ["unknown / (unknown)                 (scalar cube)"]
        rep = cube_replines(cube)
        self.assertEqual(expect, rep)
        # Shortform is the same.
        rep = cube_replines(cube, oneline=True)
        self.assertEqual(expect, rep)

    def test_name_padding(self):
        cube = Cube([1, 2], long_name="cube_accel", units="ms-2")
        rep = cube_replines(cube)
        self.assertEqual(rep, ["cube_accel / (ms-2)                 (-- : 2)"])
        rep = cube_replines(cube, name_padding=0)
        self.assertEqual(rep, ["cube_accel / (ms-2) (-- : 2)"])
        rep = cube_replines(cube, name_padding=25)
        self.assertEqual(rep, ["cube_accel / (ms-2)       (-- : 2)"])

    def test_columns_long_coordname(self):
        cube = Cube([0], long_name="short", units=1)
        coord = AuxCoord(
            [0], long_name="very_very_very_very_very_long_coord_name"
        )
        cube.add_aux_coord(coord, 0)
        rep = cube_replines(cube)
        expected = [
            "short / (1)                                      (-- : 1)",
            "    Auxiliary coordinates:",
            "        very_very_very_very_very_long_coord_name     x",
        ]
        self.assertEqual(expected, rep)
        rep = cube_replines(cube, oneline=True)
        # Note: the default short-form is short-ER, but not minimal.
        short_expected = ["short / (1)                         (-- : 1)"]
        self.assertEqual(short_expected, rep)

    def test_columns_long_attribute(self):
        cube = Cube([0], long_name="short", units=1)
        cube.attributes[
            "very_very_very_very_very_long_name"
        ] = "longish string extends beyond dim columns"
        rep = cube_replines(cube)
        expected = [
            "short / (1)                                (-- : 1)",
            "    Attributes:",
            (
                "        very_very_very_very_very_long_name "
                "'longish string extends beyond dim columns'"
            ),
        ]
        self.assertEqual(rep, expected)

    def test_coord_distinguishing_attributes(self):
        # Printout of differing attributes to differentiate same-named coords.
        # include : vector + scalar
        cube = Cube([0, 1], long_name="name", units=1)
        # Add a pair of vector coords with same name but different attributes.
        cube.add_aux_coord(
            AuxCoord([0, 1], long_name="co1", attributes=dict(a=1)), 0
        )
        cube.add_aux_coord(
            AuxCoord([0, 1], long_name="co1", attributes=dict(a=2)), 0
        )
        # Likewise for scalar coords with same name but different attributes.
        cube.add_aux_coord(
            AuxCoord([0], long_name="co2", attributes=dict(a=10, b=12))
        )
        cube.add_aux_coord(
            AuxCoord([1], long_name="co2", attributes=dict(a=10, b=11))
        )

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 2)",
            "    Auxiliary coordinates:",
            "        co1                             x",
            "            a=1",
            "        co1                             x",
            "            a=2",
            "    Scalar coordinates:",
            "        co2                         0",
            "            b=12",
            "        co2                         1",
            "            b=11",
        ]
        self.assertEqual(rep, expected)

    def test_coord_extra_attributes__array(self):
        cube = Cube(0, long_name="name", units=1)
        # Add a pair of vector coords with same name but different attributes.
        array1 = np.arange(0, 3)
        array2 = np.arange(10, 13)
        cube.add_aux_coord(
            AuxCoord([1.2], long_name="co1", attributes=dict(a=1, arr=array1))
        )
        cube.add_aux_coord(
            AuxCoord([3.4], long_name="co1", attributes=dict(a=1, arr=array2))
        )

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (scalar cube)",
            "    Scalar coordinates:",
            "        co1                         1.2",
            "            arr=array([0, 1, 2])",
            "        co1                         3.4",
            "            arr=array([10, 11, 12])",
        ]
        self.assertEqual(rep, expected)

    def test_coord_extra_attributes__array__long(self):
        # Also test with a long array representation.
        # NOTE: this also pushes the dimension map right-wards.
        array = 10 + np.arange(24.0).reshape((2, 3, 4))
        cube = Cube(0, long_name="name", units=1)
        cube.add_aux_coord(AuxCoord([1], long_name="co"))
        cube.add_aux_coord(
            AuxCoord([2], long_name="co", attributes=dict(a=array + 1.0))
        )

        rep = cube_replines(cube)
        expected = [
            (
                "name / (1)                                                 "
                "                                 (scalar cube)"
            ),
            "    Scalar coordinates:",
            (
                "        co                                                 "
                "                                 1"
            ),
            (
                "        co                                                 "
                "                                 2"
            ),
            (
                "            a=array([[[11., 12., 13., 14.], [15., 16., 17.,"
                " 18.], [19., 20., 21., 22.]],..."
            ),
        ]
        self.assertEqual(rep, expected)

    def test_coord_extra_attributes__string(self):
        cube = Cube(0, long_name="name", units=1)
        cube.add_aux_coord(AuxCoord([1], long_name="co"))
        cube.add_aux_coord(
            AuxCoord(
                [2], long_name="co", attributes=dict(note="string content")
            )
        )
        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (scalar cube)",
            "    Scalar coordinates:",
            "        co                          1",
            "        co                          2",
            "            note='string content'",
        ]
        self.assertEqual(rep, expected)

    def test_coord_extra_attributes__string_escaped(self):
        cube = Cube(0, long_name="name", units=1)
        cube.add_aux_coord(AuxCoord([1], long_name="co"))
        cube.add_aux_coord(
            AuxCoord(
                [2],
                long_name="co",
                attributes=dict(note="line 1\nline 2\tends."),
            )
        )
        rep = cube_replines(cube)
        expected = [
            "name / (1)                               (scalar cube)",
            "    Scalar coordinates:",
            "        co                               1",
            "        co                               2",
            "            note='line 1\\nline 2\\tends.'",
        ]
        self.assertEqual(rep, expected)

    def test_coord_extra_attributes__string_overlong(self):
        cube = Cube(0, long_name="name", units=1)
        cube.add_aux_coord(AuxCoord([1], long_name="co"))
        long_string = (
            "this is very very very very very very very "
            "very very very very very very very long."
        )
        cube.add_aux_coord(
            AuxCoord([2], long_name="co", attributes=dict(note=long_string))
        )
        rep = cube_replines(cube)
        expected = [
            (
                "name / (1)                                    "
                "                                           (scalar cube)"
            ),
            "    Scalar coordinates:",
            (
                "        co                                    "
                "                                           1"
            ),
            (
                "        co                                    "
                "                                           2"
            ),
            (
                "            note='this is very very very very "
                "very very very very very very very very..."
            ),
        ]
        self.assertEqual(rep, expected)

    def test_section_vector_dimcoords(self):
        cube = Cube(np.zeros((2, 3)), long_name="name", units=1)
        cube.add_dim_coord(DimCoord([0, 1], long_name="y"), 0)
        cube.add_dim_coord(DimCoord([0, 1, 2], long_name="x"), 1)

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (y: 2; x: 3)",
            "    Dimension coordinates:",
            "        y                             x     -",
            "        x                             -     x",
        ]
        self.assertEqual(rep, expected)

    def test_section_vector_auxcoords(self):
        cube = Cube(np.zeros((2, 3)), long_name="name", units=1)
        cube.add_aux_coord(DimCoord([0, 1], long_name="y"), 0)
        cube.add_aux_coord(DimCoord([0, 1, 2], long_name="x"), 1)

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 2; -- : 3)",
            "    Auxiliary coordinates:",
            "        y                               x       -",
            "        x                               -       x",
        ]
        self.assertEqual(rep, expected)

    def test_section_vector_ancils(self):
        cube = Cube(np.zeros((2, 3)), long_name="name", units=1)
        cube.add_ancillary_variable(
            AncillaryVariable([0, 1], long_name="av1"), 0
        )

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 2; -- : 3)",
            "    Ancillary variables:",
            "        av1                             x       -",
        ]
        self.assertEqual(rep, expected)

    def test_section_vector_ancils_length_1(self):
        # Check ancillary variables that map to a cube dimension of length 1
        # are not interpreted as scalar ancillary variables.
        cube = Cube(np.zeros((1, 3)), long_name="name", units=1)
        cube.add_ancillary_variable(AncillaryVariable([0], long_name="av1"), 0)

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 1; -- : 3)",
            "    Ancillary variables:",
            "        av1                             x       -",
        ]
        self.assertEqual(rep, expected)

    def test_section_vector_cell_measures(self):
        cube = Cube(np.zeros((2, 3)), long_name="name", units=1)
        cube.add_cell_measure(CellMeasure([0, 1, 2], long_name="cm"), 1)

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 2; -- : 3)",
            "    Cell measures:",
            "        cm                              -       x",
        ]
        self.assertEqual(rep, expected)

    def test_section_vector_cell_measures_length_1(self):
        # Check cell measures that map to a cube dimension of length 1 are not
        # interpreted as scalar cell measures.
        cube = Cube(np.zeros((2, 1)), long_name="name", units=1)
        cube.add_cell_measure(CellMeasure([0], long_name="cm"), 1)

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 2; -- : 1)",
            "    Cell measures:",
            "        cm                              -       x",
        ]
        self.assertEqual(rep, expected)

    def test_section_scalar_coords(self):
        # incl points + bounds
        # TODO: ought to incorporate coord-based summary
        #  - which would allow for special printout of time values
        cube = Cube([0], long_name="name", units=1)
        cube.add_aux_coord(DimCoord([0.0], long_name="unbounded"))
        cube.add_aux_coord(DimCoord([0], bounds=[[0, 7]], long_name="bounded"))

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 1)",
            "    Scalar coordinates:",
            "        bounded                     0, bound=(0, 7)",
            "        unbounded                   0.0",
        ]
        self.assertEqual(rep, expected)

    def test_section_scalar_coords__string(self):
        # incl a newline-escaped one
        # incl a long (clipped) one
        # CHECK THAT CLIPPED+ESCAPED WORKS (don't lose final quote)
        cube = Cube([0], long_name="name", units=1)
        cube.add_aux_coord(AuxCoord(["string-value"], long_name="text"))
        long_string = (
            "A string value which is very very very very very very "
            "very very very very very very very very long."
        )
        cube.add_aux_coord(
            AuxCoord([long_string], long_name="very_long_string")
        )

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 1)",
            "    Scalar coordinates:",
            "        text                        string-value",
            (
                "        very_long_string            A string value which is "
                "very very very very very very very very very very..."
            ),
        ]
        self.assertEqual(rep, expected)

    def test_section_scalar_cell_measures(self):
        cube = Cube(np.zeros((2, 3)), long_name="name", units=1)
        cube.add_cell_measure(CellMeasure([0], long_name="cm"))

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 2; -- : 3)",
            "    Scalar cell measures:",
            "        cm",
        ]
        self.assertEqual(rep, expected)

    def test_section_scalar_ancillaries(self):
        # There *is* no section for this.  But there probably ought to be.
        cube = Cube(np.zeros((2, 3)), long_name="name", units=1)
        cube.add_ancillary_variable(AncillaryVariable([0], long_name="av"))

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 2; -- : 3)",
            "    Scalar ancillary variables:",
            "        av",
        ]
        self.assertEqual(rep, expected)

    def test_section_cube_attributes(self):
        cube = Cube([0], long_name="name", units=1)
        cube.attributes["number"] = 1.2
        cube.attributes["list"] = [3]
        cube.attributes["string"] = "four five in a string"
        cube.attributes["z_tupular"] = (6, (7, 8))
        rep = cube_replines(cube)
        # NOTE: 'list' before 'number', as it uses "sorted(attrs.items())"
        expected = [
            "name / (1)                          (-- : 1)",
            "    Attributes:",
            "        list                        [3]",
            "        number                      1.2",
            "        string                      'four five in a string'",
            "        z_tupular                   (6, (7, 8))",
        ]
        self.assertEqual(rep, expected)

    def test_section_cube_attributes__string_extras(self):
        cube = Cube([0], long_name="name", units=1)
        # Overlong strings are truncated (with iris.util.clip_string).
        long_string = (
            "this is very very very very very very very "
            "very very very very very very very long."
        )
        # Strings with embedded newlines or quotes are printed in quoted form.
        cube.attributes["escaped"] = "escaped\tstring"
        cube.attributes["long"] = long_string
        cube.attributes["long_multi"] = "multi\nline, " + long_string
        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 1)",
            "    Attributes:",
            "        escaped                     'escaped\\tstring'",
            (
                "        long                        'this is very very very "
                "very very very very very very very very very very ...'"
            ),
            (
                "        long_multi                  'multi\\nline, "
                "this is very very very very very very very very very very ...'"
            ),
        ]
        self.assertEqual(rep, expected)

    def test_section_cube_attributes__array(self):
        # Including  a long one, which gets a truncated representation.
        cube = Cube([0], long_name="name", units=1)
        small_array = np.array([1.2, 3.4])
        large_array = np.arange(36).reshape((18, 2))
        cube.attributes["array"] = small_array
        cube.attributes["bigarray"] = large_array
        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (-- : 1)",
            "    Attributes:",
            "        array                       array([1.2, 3.4])",
            (
                "        bigarray                    array([[ 0, 1], [ 2, 3], "
                "[ 4, 5], [ 6, 7], [ 8, 9], [10, 11], [12, 13], ..."
            ),
        ]
        self.assertEqual(rep, expected)

    def test_section_cell_methods(self):
        cube = Cube([0], long_name="name", units=1)
        cube.add_cell_method(CellMethod("stdev", "area"))
        cube.add_cell_method(
            CellMethod(
                method="mean",
                coords=["y", "time"],
                intervals=["10m", "3min"],
                comments=["vertical", "=duration"],
            )
        )
        rep = cube_replines(cube)
        # Note: not alphabetical -- provided order is significant
        expected = [
            "name / (1)                          (-- : 1)",
            "    Cell methods:",
            "        stdev                       area",
            "        mean                        y (10m, vertical), time (3min, =duration)",
        ]
        self.assertEqual(rep, expected)

    def test_unstructured_cube(self):
        # Check a sample mesh-cube against the expected result.
        cube = sample_mesh_cube()
        rep = cube_replines(cube)
        expected = [
            "mesh_phenom / (unknown)             (level: 2; i_mesh_face: 3)",
            "    Dimension coordinates:",
            "        level                             x               -",
            "        i_mesh_face                       -               x",
            "    Mesh coordinates:",
            "        latitude                          -               x",
            "        longitude                         -               x",
            "    Auxiliary coordinates:",
            "        mesh_face_aux                     -               x",
            "    Mesh:",
            "        name                        unknown",
            "        location                    face",
        ]
        self.assertEqual(rep, expected)


if __name__ == "__main__":
    tests.main()
