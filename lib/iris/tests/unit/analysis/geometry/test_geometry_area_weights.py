# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris.analysis.geometry.geometry_area_weights`
function.

 """

# Import iris.tests first so that some things can be initialised before
# importing anything else.

import iris.tests as tests  # isort:skip
import warnings

import numpy as np
import shapely.geometry

from iris.analysis.geometry import geometry_area_weights
from iris.coords import DimCoord
from iris.cube import Cube
import iris.tests.stock as stock


class Test(tests.IrisTest):
    def setUp(self):
        x_coord = DimCoord([1.0, 3.0], "longitude", bounds=[[0, 2], [2, 4]])
        y_coord = DimCoord([1.0, 3.0], "latitude", bounds=[[0, 2], [2, 4]])
        self.data = np.empty((4, 2, 2))
        dim_coords_and_dims = [(y_coord, (1,)), (x_coord, (2,))]
        self.cube = Cube(self.data, dim_coords_and_dims=dim_coords_and_dims)
        self.geometry = shapely.geometry.Polygon(
            [(3, 3), (3, 50), (50, 50), (50, 3)]
        )

    def test_no_overlap(self):
        geometry = shapely.geometry.Polygon([(4, 4), (4, 6), (6, 6), (6, 4)])
        weights = geometry_area_weights(self.cube, geometry)
        self.assertEqual(np.sum(weights), 0)

    def test_overlap(self):
        weights = geometry_area_weights(self.cube, self.geometry)
        expected = np.repeat(
            [[[0.0, 0.0], [0.0, 1.0]]], self.data.shape[0], axis=0
        )
        self.assertArrayEqual(weights, expected)

    def test_overlap_normalize(self):
        weights = geometry_area_weights(
            self.cube, self.geometry, normalize=True
        )
        expected = np.repeat(
            [[[0.0, 0.0], [0.0, 0.25]]], self.data.shape[0], axis=0
        )
        self.assertArrayEqual(weights, expected)

    @tests.skip_data
    def test_distinct_xy(self):
        cube = stock.simple_pp()
        cube = cube[:4, :4]
        lon = cube.coord("longitude")
        lat = cube.coord("latitude")
        lon.guess_bounds()
        lat.guess_bounds()
        from iris.util import regular_step

        quarter = abs(regular_step(lon) * regular_step(lat) * 0.25)
        half = abs(regular_step(lon) * regular_step(lat) * 0.5)
        minx = 3.7499990463256836
        maxx = 7.499998092651367
        miny = 84.99998474121094
        maxy = 89.99998474121094
        geometry = shapely.geometry.box(minx, miny, maxx, maxy)
        weights = geometry_area_weights(cube, geometry)
        target = np.array(
            [
                [0, quarter, quarter, 0],
                [0, half, half, 0],
                [0, quarter, quarter, 0],
                [0, 0, 0, 0],
            ]
        )
        self.assertTrue(np.allclose(weights, target))

    @tests.skip_data
    def test_distinct_xy_bounds(self):
        # cases where geometry bnds are outside cube bnds correctly handled?
        cube = stock.simple_pp()
        cube = cube[:4, :4]
        lon = cube.coord("longitude")
        lat = cube.coord("latitude")
        lon.guess_bounds()
        lat.guess_bounds()
        from iris.util import regular_step

        quarter = abs(regular_step(lon) * regular_step(lat) * 0.25)
        half = abs(regular_step(lon) * regular_step(lat) * 0.5)
        full = abs(regular_step(lon) * regular_step(lat))
        minx = 3.7499990463256836
        maxx = 13.12499619
        maxx_overshoot = 15.0
        miny = 84.99998474121094
        maxy = 89.99998474121094
        geometry = shapely.geometry.box(minx, miny, maxx, maxy)
        geometry_overshoot = shapely.geometry.box(
            minx, miny, maxx_overshoot, maxy
        )
        weights = geometry_area_weights(cube, geometry)
        weights_overshoot = geometry_area_weights(cube, geometry_overshoot)
        target = np.array(
            [
                [0, quarter, half, half],
                [0, half, full, full],
                [0, quarter, half, half],
                [0, 0, 0, 0],
            ]
        )
        self.assertTrue(np.allclose(weights, target))
        self.assertTrue(np.allclose(weights_overshoot, target))

    @tests.skip_data
    def test_distinct_xy_bounds_pole(self):
        # is UserWarning issued for out-of-bounds? results will be unexpected!
        cube = stock.simple_pp()
        cube = cube[:4, :4]
        lon = cube.coord("longitude")
        lat = cube.coord("latitude")
        lon.guess_bounds()
        lat.guess_bounds()
        from iris.util import regular_step

        quarter = abs(regular_step(lon) * regular_step(lat) * 0.25)
        half = abs(regular_step(lon) * regular_step(lat) * 0.5)
        top_cell_half = abs(regular_step(lon) * (90 - lat.bounds[0, 1]) * 0.5)
        minx = 3.7499990463256836
        maxx = 7.499998092651367
        miny = 84.99998474121094
        maxy = 99.99998474121094
        geometry = shapely.geometry.box(minx, miny, maxx, maxy)
        # see http://stackoverflow.com/a/3892301 to assert warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # always trigger all warnings
            weights = geometry_area_weights(cube, geometry)
            self.assertEqual(
                str(w[-1].message),
                "The geometry exceeds the "
                "cube's y dimension at the upper end.",
            )
            self.assertTrue(issubclass(w[-1].category, UserWarning))
        target = np.array(
            [
                [0, top_cell_half, top_cell_half, 0],
                [0, half, half, 0],
                [0, quarter, quarter, 0],
                [0, 0, 0, 0],
            ]
        )
        self.assertTrue(np.allclose(weights, target))

    def test_shared_xy(self):
        cube = stock.track_1d()
        geometry = shapely.geometry.box(1, 4, 3.5, 7)
        weights = geometry_area_weights(cube, geometry)
        target = np.array([0, 0, 2, 0.5, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.allclose(weights, target))


if __name__ == "__main__":
    tests.main()
