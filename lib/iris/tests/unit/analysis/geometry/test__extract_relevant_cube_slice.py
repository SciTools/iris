# (C) British Crown Copyright 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Unit tests for :func:`iris.analysis.geometry._extract_relevant_cube_slice`.

"""

from __future__ import division
# Import iris.tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests
import iris.tests.stock

import numpy as np
import shapely.geometry

import iris
from iris.analysis.geometry import _extract_relevant_cube_slice


class Test(tests.IrisTest):
    def test_polygon_smaller_than_cube(self):
        cube = tests.stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.box(-0.4, -0.4, 0.4, 0.4)
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (cube[1, 1],
                  cube[1, 1].coords(axis='x')[0],
                  cube[1, 1].coords(axis='y')[0],
                  (1, 1, 1, 1))
        self.assertEqual(target, actual)

    def test_polygon_larger_than_cube(self):
        cube = tests.stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.box(-0.6, -0.6, 0.6, 0.6)
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (cube[:, :3],
                  cube[:, :3].coords(axis='x')[0],
                  cube[:, :3].coords(axis='y')[0],
                  (0, 0, 2, 2))
        self.assertEqual(target, actual)

    def test_polygon_on_cube_boundary(self):
        cube = tests.stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.box(-0.5, -0.5, 0.5, 0.5)
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (cube[1, 1],
                  cube[1, 1].coords(axis='x')[0],
                  cube[1, 1].coords(axis='y')[0],
                  (1, 1, 1, 1))
        self.assertEqual(target, actual)

    def test_rotated_polygon_on_cube_boundary(self):
        cube = tests.stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.Polygon(((0., -.5), (-.5, 0.), (0., .5),
                                             (.5, 0.)))
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (cube[1, 1],
                  cube[1, 1].coords(axis='x')[0],
                  cube[1, 1].coords(axis='y')[0],
                  (1, 1, 1, 1))
        self.assertEqual(target, actual)

    def test_rotated_polygon_larger_than_cube_boundary(self):
        cube = tests.stock.lat_lon_cube()
        cube.dim_coords[0].guess_bounds()
        cube.dim_coords[1].guess_bounds()
        geometry = shapely.geometry.Polygon(((0., -.6), (-.6, 0.), (0., .6),
                                             (.6, 0.)))
        actual = _extract_relevant_cube_slice(cube, geometry)
        target = (cube[:, :3],
                  cube[:, :3].coords(axis='x')[0],
                  cube[:, :3].coords(axis='y')[0],
                  (0, 0, 2, 2))
        self.assertEqual(target, actual)


if __name__ == "__main__":
    tests.main()
