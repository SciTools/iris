# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris._representation.cube_printout.CubePrintout`."""
import iris.tests as tests

import numpy as np

from iris.cube import Cube
from iris.coords import AuxCoord, DimCoord
from iris._representation.cube_summary import CubeSummary

from iris._representation.cube_printout import CubePrinter


def cube_replines(cube, **kwargs):
    return CubePrinter(CubeSummary(cube)).to_string(**kwargs).split("\n")


class TestCubePrintout__to_string(tests.IrisTest):
    def test_empty(self):
        cube = Cube([0])
        rep = cube_replines(cube)
        self.assertEqual(rep, ["unknown / (unknown)                 (-- : 1)"])
        rep = cube_replines(cube, oneline=True)
        self.assertEqual(rep, ["unknown / (unknown) (-- : 1)"])

    def test_scalar_cube_summaries(self):
        cube = Cube(0)
        rep = cube_replines(cube)
        self.assertEqual(
            rep, ["unknown / (unknown)                 (scalar cube)"]
        )
        rep = cube_replines(cube, oneline=True)
        self.assertEqual(rep, ["unknown / (unknown) (scalar cube)"])

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
            "        very_very_very_very_very_long_coord_name       x",
        ]
        self.assertEqual(rep, expected)
        rep = cube_replines(cube, oneline=True)
        self.assertEqual(rep, ["short / (1) (-- : 1)"])

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
                "longish string extends beyond dim columns"
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
            "        co1                               x",
            "            a=1",
            "        co1                               x",
            "            a=2",
            "    Scalar coordinates:",
            "        co2                         0",
            "            b=12",
            "        co2                         1",
            "            b=11",
        ]
        self.assertEqual(rep, expected)

    def test_coord_extra_attributes__array(self):
        # Include : long
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
                attributes=dict(note="line 1\nline 2\nends."),
            )
        )
        rep = cube_replines(cube)
        expected = [
            "name / (1)                               (scalar cube)",
            "    Scalar coordinates:",
            "        co                               1",
            "        co                               2",
            "            note='line 1\\nline 2\\nends.'",
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
        # Add a pair of vector coords with same name but different attributes.
        cube.add_dim_coord(DimCoord([0, 1], long_name="y"), 0)
        cube.add_dim_coord(DimCoord([0, 1, 2], long_name="x"), 1)

        rep = cube_replines(cube)
        expected = [
            "name / (1)                          (y: 2; x: 3)",
            "    Dimension coordinates:",
            "        y                               x     -",
            "        x                               -     x",
        ]
        self.assertEqual(rep, expected)

    def test_section_vector_auxcoords(self):
        pass

    def test_section_vector_ancils(self):
        pass

    def test_section_vector_cell_measures(self):
        pass

    def test_section_scalar_coords(self):
        # incl points + bounds
        # TODO: ought to incorporate coord-based summary
        #  - which would allow for special printout of time values
        pass

    def test_section_scalar_coords__string(self):
        # incl a newline-escaped one
        # incl a long (clipped) one
        # CHECK THAT CLIPPED+ESCAPED WORKS (don't lose final quote)
        pass

    def test_section_scalar_cell_measures(self):
        pass

    def test_section_scalar_cube_attributes(self):
        pass

    def test_section_cube_attributes__string(self):
        # incl a newline-escaped one
        # incl a long (clipped) one
        # CHECK THAT CLIPPED+ESCAPED WORKS (don't lose final quote)
        pass

    def test_section_cube_attributes__array(self):
        # incl a long one
        pass


if __name__ == "__main__":
    tests.main()
