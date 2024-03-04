# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.analysis.geometry._extract_relevant_cube_slice`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests  # isort:skip
import shapely.geometry

from iris.analysis.geometry import _extract_relevant_cube_slice
import iris.tests.stock as stock


class Test(tests.IrisTest):
    def test_polygon_smaller_than_cube(self):
        cube = stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.box(-0.4, -0.4, 0.4, 0.4)
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (
            cube[1, 1],
            cube[1, 1].coords(axis="x")[0],
            cube[1, 1].coords(axis="y")[0],
            (1, 1, 1, 1),
        )
        self.assertEqual(target, actual)

    def test_polygon_larger_than_cube(self):
        cube = stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.box(-0.6, -0.6, 0.6, 0.6)
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (
            cube[:, :3],
            cube[:, :3].coords(axis="x")[0],
            cube[:, :3].coords(axis="y")[0],
            (0, 0, 2, 2),
        )
        self.assertEqual(target, actual)

    def test_polygon_on_cube_boundary(self):
        cube = stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.box(-0.5, -0.5, 0.5, 0.5)
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (
            cube[1, 1],
            cube[1, 1].coords(axis="x")[0],
            cube[1, 1].coords(axis="y")[0],
            (1, 1, 1, 1),
        )
        self.assertEqual(target, actual)

    def test_rotated_polygon_on_cube_boundary(self):
        cube = stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.Polygon(
            ((0.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 0.0))
        )
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (
            cube[1, 1],
            cube[1, 1].coords(axis="x")[0],
            cube[1, 1].coords(axis="y")[0],
            (1, 1, 1, 1),
        )
        self.assertEqual(target, actual)

    def test_rotated_polygon_larger_than_cube_boundary(self):
        cube = stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.Polygon(
            ((0.0, -0.6), (-0.6, 0.0), (0.0, 0.6), (0.6, 0.0))
        )
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (
            cube[:, :3],
            cube[:, :3].coords(axis="x")[0],
            cube[:, :3].coords(axis="y")[0],
            (0, 0, 2, 2),
        )
        self.assertEqual(target, actual)


if __name__ == "__main__":
    tests.main()
