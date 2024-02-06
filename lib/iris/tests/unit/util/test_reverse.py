# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.reverse`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import unittest

import numpy as np

import iris
from iris.util import reverse


class Test_array(tests.IrisTest):
    def test_simple_array(self):
        a = np.arange(12).reshape(3, 4)
        self.assertArrayEqual(a[::-1], reverse(a, 0))
        self.assertArrayEqual(a[::-1, ::-1], reverse(a, [0, 1]))
        self.assertArrayEqual(a[:, ::-1], reverse(a, 1))
        self.assertArrayEqual(a[:, ::-1], reverse(a, [1]))

        msg = "Reverse was expecting a single axis or a 1d array *"
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, [])

        msg = "An axis value out of range for the number of dimensions *"
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, -1)
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, 10)
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, [-1])
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, [0, -1])

        msg = "To reverse an array, provide an int *"
        with self.assertRaisesRegex(TypeError, msg):
            reverse(a, "latitude")

    def test_single_array(self):
        a = np.arange(36).reshape(3, 4, 3)
        self.assertArrayEqual(a[::-1], reverse(a, 0))
        self.assertArrayEqual(a[::-1, ::-1], reverse(a, [0, 1]))
        self.assertArrayEqual(a[:, ::-1, ::-1], reverse(a, [1, 2]))
        self.assertArrayEqual(a[..., ::-1], reverse(a, 2))

        msg = "Reverse was expecting a single axis or a 1d array *"
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, [])

        msg = "An axis value out of range for the number of dimensions *"
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, -1)
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, 10)
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, [-1])
        with self.assertRaisesRegex(ValueError, msg):
            reverse(a, [0, -1])

        with self.assertRaisesRegex(TypeError, "To reverse an array, provide an int *"):
            reverse(a, "latitude")


class Test_cube(tests.IrisTest):
    def setUp(self):
        # On this cube pair, the coordinates to perform operations on have
        # matching long names but the points array on one cube is reversed
        # with respect to that on the other.
        data = np.arange(12).reshape(3, 4)

        self.a1 = iris.coords.DimCoord([1, 2, 3], long_name="a")
        self.a1.guess_bounds()
        self.b1 = iris.coords.DimCoord([1, 2, 3, 4], long_name="b")

        a2 = iris.coords.DimCoord([3, 2, 1], long_name="a")
        a2.guess_bounds()
        b2 = iris.coords.DimCoord([4, 3, 2, 1], long_name="b")

        self.span = iris.coords.AuxCoord(
            np.arange(12).reshape(3, 4), long_name="spanning"
        )

        self.cube1 = iris.cube.Cube(
            data,
            dim_coords_and_dims=[(self.a1, 0), (self.b1, 1)],
            aux_coords_and_dims=[(self.span, (0, 1))],
        )

        self.cube2 = iris.cube.Cube(data, dim_coords_and_dims=[(a2, 0), (b2, 1)])

    def check_coorda_reversed(self, result):
        self.assertArrayEqual(self.cube2.coord("a").points, result.coord("a").points)
        self.assertArrayEqual(self.cube2.coord("a").bounds, result.coord("a").bounds)

    def check_coorda_unchanged(self, result):
        self.assertArrayEqual(self.cube1.coord("a").points, result.coord("a").points)
        self.assertArrayEqual(self.cube1.coord("a").bounds, result.coord("a").bounds)

    def check_coordb_reversed(self, result):
        self.assertArrayEqual(self.cube2.coord("b").points, result.coord("b").points)

    def check_coordb_unchanged(self, result):
        self.assertArrayEqual(self.cube1.coord("b").points, result.coord("b").points)

    def test_cube_dim0(self):
        cube1_reverse0 = reverse(self.cube1, 0)

        self.assertArrayEqual(self.cube1.data[::-1], cube1_reverse0.data)
        self.check_coorda_reversed(cube1_reverse0)
        self.check_coordb_unchanged(cube1_reverse0)

    def test_cube_dim1(self):
        cube1_reverse1 = reverse(self.cube1, 1)

        self.assertArrayEqual(self.cube1.data[:, ::-1], cube1_reverse1.data)
        self.check_coordb_reversed(cube1_reverse1)
        self.check_coorda_unchanged(cube1_reverse1)

    def test_cube_dim_both(self):
        cube1_reverse_both = reverse(self.cube1, (0, 1))

        self.assertArrayEqual(self.cube1.data[::-1, ::-1], cube1_reverse_both.data)
        self.check_coorda_reversed(cube1_reverse_both)
        self.check_coordb_reversed(cube1_reverse_both)

    def test_cube_coord0(self):
        cube1_reverse0 = reverse(self.cube1, self.a1)

        self.assertArrayEqual(self.cube1.data[::-1], cube1_reverse0.data)
        self.check_coorda_reversed(cube1_reverse0)
        self.check_coordb_unchanged(cube1_reverse0)

    def test_cube_coord1(self):
        cube1_reverse1 = reverse(self.cube1, "b")

        self.assertArrayEqual(self.cube1.data[:, ::-1], cube1_reverse1.data)
        self.check_coordb_reversed(cube1_reverse1)
        self.check_coorda_unchanged(cube1_reverse1)

    def test_cube_coord_both(self):
        cube1_reverse_both = reverse(self.cube1, (self.a1, self.b1))

        self.assertArrayEqual(self.cube1.data[::-1, ::-1], cube1_reverse_both.data)
        self.check_coorda_reversed(cube1_reverse_both)
        self.check_coordb_reversed(cube1_reverse_both)

    def test_cube_coord_spanning(self):
        cube1_reverse_spanning = reverse(self.cube1, "spanning")

        self.assertArrayEqual(self.cube1.data[::-1, ::-1], cube1_reverse_spanning.data)
        self.check_coorda_reversed(cube1_reverse_spanning)
        self.check_coordb_reversed(cube1_reverse_spanning)

        self.assertArrayEqual(
            self.span.points[::-1, ::-1],
            cube1_reverse_spanning.coord("spanning").points,
        )

    def test_wrong_coord_name(self):
        msg = "Expected to find exactly 1 'latitude' coordinate, but found none."
        with self.assertRaisesRegex(iris.exceptions.CoordinateNotFoundError, msg):
            reverse(self.cube1, "latitude")

    def test_empty_list(self):
        msg = "Reverse was expecting a single axis or a 1d array *"
        with self.assertRaisesRegex(ValueError, msg):
            reverse(self.cube1, [])

    def test_wrong_type_cube(self):
        msg = (
            "coords_or_dims must be int, str, coordinate or sequence of "
            "these.  Got cube."
        )
        with self.assertRaisesRegex(TypeError, msg):
            reverse(self.cube1, self.cube1)

    def test_wrong_type_float(self):
        msg = "coords_or_dims must be int, str, coordinate or sequence of these."
        with self.assertRaisesRegex(TypeError, msg):
            reverse(self.cube1, 3.0)


if __name__ == "__main__":
    unittest.main()
