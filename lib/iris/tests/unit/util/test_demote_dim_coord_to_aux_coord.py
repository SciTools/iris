# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :func:`iris.util.demote_dim_coord_to_aux_coord`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import unittest

import iris
import iris.tests.stock as stock
from iris.util import demote_dim_coord_to_aux_coord


class Test(tests.IrisTest):
    def test_argument_is_basestring(self):
        cube_a = stock.simple_3d()
        cube_b = cube_a.copy()
        demote_dim_coord_to_aux_coord(cube_b, cube_b.coord("wibble"))
        self.assertEqual(
            cube_b.dim_coords,
            (cube_a.coord("latitude"), cube_a.coord("longitude")),
        )

    @tests.skip_data
    def test_argument_is_coord_instance(self):
        cube_a = stock.realistic_4d()
        cube_b = cube_a.copy()
        coord = cube_b.coord("model_level_number").copy()
        demote_dim_coord_to_aux_coord(cube_b, coord)
        self.assertEqual(
            cube_b.dim_coords,
            (
                cube_a.coord("time"),
                cube_a.coord("grid_latitude"),
                cube_a.coord("grid_longitude"),
            ),
        )

    def test_old_dim_coord_is_now_aux_coord(self):
        cube_a = stock.hybrid_height()
        cube_b = cube_a.copy()
        demote_dim_coord_to_aux_coord(cube_b, "level_height")
        self.assertTrue(cube_a.coord("level_height") in cube_b.aux_coords)

    def test_coord_of_that_name_does_not_exist(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            demote_dim_coord_to_aux_coord(cube_a, "wibble")

    def test_coord_does_not_exist(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        cube_b = cube_a.copy()
        coord = cube_b.coord("dim1").copy()
        coord.rename("new")
        demote_dim_coord_to_aux_coord(cube_b, coord)
        self.assertEqual(cube_a, cube_b)

    def test_argument_is_wrong_type(self):
        cube_a = stock.simple_1d()
        with self.assertRaises(TypeError):
            demote_dim_coord_to_aux_coord(cube_a, 0.0)

    def test_trying_to_demote_a_scalar_coord(self):
        cube_a = stock.simple_2d_w_multidim_and_scalars()
        cube_b = cube_a.copy()
        demote_dim_coord_to_aux_coord(cube_b, "an_other")
        self.assertEqual(cube_a, cube_b)


if __name__ == "__main__":
    unittest.main()
