# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:meth:`iris.analysis.trajectory._nearest_neighbour_indices_ndcoords`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.analysis.trajectory import (
    _nearest_neighbour_indices_ndcoords as nn_ndinds,
)
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


class Test2d(tests.IrisTest):
    def test_nonlatlon_simple_2d(self):
        co_y = DimCoord([10.0, 20.0], long_name="y")
        co_x = DimCoord([1.0, 2.0, 3.0], long_name="x")
        cube = Cube(np.zeros((2, 3)))
        cube.add_dim_coord(co_y, 0)
        cube.add_dim_coord(co_x, 1)
        sample_point = [("x", 2.8), ("y", 18.5)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(1, 2)])

    def test_nonlatlon_multiple_2d(self):
        co_y = DimCoord([10.0, 20.0], long_name="y")
        co_x = DimCoord([1.0, 2.0, 3.0], long_name="x")
        cube = Cube(np.zeros((2, 3)))
        cube.add_dim_coord(co_y, 0)
        cube.add_dim_coord(co_x, 1)
        sample_points = [("x", [2.8, -350.0, 1.7]), ("y", [18.5, 8.7, 12.2])]
        result = nn_ndinds(cube, sample_points)
        self.assertEqual(result, [(1, 2), (0, 0), (0, 1)])

    def test_latlon_simple_2d(self):
        co_y = DimCoord(
            [10.0, 20.0], standard_name="latitude", units="degrees"
        )
        co_x = DimCoord(
            [1.0, 2.0, 3.0], standard_name="longitude", units="degrees"
        )
        cube = Cube(np.zeros((2, 3)))
        cube.add_dim_coord(co_y, 0)
        cube.add_dim_coord(co_x, 1)
        sample_point = [("longitude", 2.8), ("latitude", 18.5)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(1, 2)])

    def test_latlon_multiple_2d(self):
        co_y = DimCoord(
            [10.0, 20.0], standard_name="latitude", units="degrees"
        )
        co_x = DimCoord(
            [1.0, 2.0, 3.0], standard_name="longitude", units="degrees"
        )
        cube = Cube(np.zeros((2, 3)))
        cube.add_dim_coord(co_y, 0)
        cube.add_dim_coord(co_x, 1)
        sample_points = [
            ("longitude", [2.8, -350.0, 1.7]),
            ("latitude", [18.5, 8.7, 12.2]),
        ]
        result = nn_ndinds(cube, sample_points)
        # Note slight difference from non-latlon version.
        self.assertEqual(result, [(1, 2), (0, 2), (0, 1)])


class Test1d(tests.IrisTest):
    def test_nonlatlon_simple_1d(self):
        co_x = AuxCoord([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], long_name="x")
        co_y = AuxCoord([10.0, 10.0, 10.0, 20.0, 20.0, 20.0], long_name="y")
        cube = Cube(np.zeros(6))
        cube.add_aux_coord(co_y, 0)
        cube.add_aux_coord(co_x, 0)
        sample_point = [("x", 2.8), ("y", 18.5)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(5,)])

    def test_latlon_simple_1d(self):
        cube = Cube([11.0, 12.0, 13.0, 21.0, 22.0, 23.0])
        co_x = AuxCoord(
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            standard_name="longitude",
            units="degrees",
        )
        co_y = AuxCoord(
            [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            standard_name="latitude",
            units="degrees",
        )
        cube.add_aux_coord(co_y, 0)
        cube.add_aux_coord(co_x, 0)
        sample_point = [("longitude", 2.8), ("latitude", 18.5)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(5,)])


class TestApiExtras(tests.IrisTest):
    # Check operation with alternative calling setups.
    def test_no_y_dim(self):
        # Operate in X only, returned slice should be [:, ix].
        co_x = DimCoord([1.0, 2.0, 3.0], long_name="x")
        co_y = DimCoord([10.0, 20.0], long_name="y")
        cube = Cube(np.zeros((2, 3)))
        cube.add_dim_coord(co_y, 0)
        cube.add_dim_coord(co_x, 1)
        sample_point = [("x", 2.8)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(slice(None), 2)])

    def test_no_x_dim(self):
        # Operate in Y only, returned slice should be [iy, :].
        co_x = DimCoord([1.0, 2.0, 3.0], long_name="x")
        co_y = DimCoord([10.0, 20.0], long_name="y")
        cube = Cube(np.zeros((2, 3)))
        cube.add_dim_coord(co_y, 0)
        cube.add_dim_coord(co_x, 1)
        sample_point = [("y", 18.5)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(1, slice(None))])

    def test_sample_dictionary(self):
        # Pass sample_point arg as a dictionary: this usage mode is deprecated.
        co_x = AuxCoord([1.0, 2.0, 3.0], long_name="x")
        co_y = AuxCoord([10.0, 20.0], long_name="y")
        cube = Cube(np.zeros((2, 3)))
        cube.add_aux_coord(co_y, 0)
        cube.add_aux_coord(co_x, 1)
        sample_point = {"x": 2.8, "y": 18.5}
        exp_emsg = r"must be a list of \(coordinate, value\) pairs"
        with self.assertRaisesRegex(TypeError, exp_emsg):
            nn_ndinds(cube, sample_point)


class TestLatlon(tests.IrisTest):
    # Check correct calculations on lat-lon points.
    def _testcube_latlon_1d(self, lats, lons):
        cube = Cube(np.zeros(len(lons)))
        co_x = AuxCoord(lons, standard_name="longitude", units="degrees")
        co_y = AuxCoord(lats, standard_name="latitude", units="degrees")
        cube.add_aux_coord(co_y, 0)
        cube.add_aux_coord(co_x, 0)
        return cube

    def _check_latlon_1d(self, lats, lons, sample_point, expect):
        cube = self._testcube_latlon_1d(lats, lons)
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(expect,)])

    def test_lat_scaling(self):
        # Check that (88, 25) is closer to (88, 0) than to (87, 25)
        self._check_latlon_1d(
            lats=[88, 87],
            lons=[0, 25],
            sample_point=[("latitude", 88), ("longitude", 25)],
            expect=0,
        )

    def test_alternate_latlon_names_okay(self):
        # Check that (88, 25) is **STILL** closer to (88, 0) than to (87, 25)
        # ... when coords have odd, but still recognisable, latlon names.
        cube = self._testcube_latlon_1d(lats=[88, 87], lons=[0, 25])
        cube.coord("latitude").rename("y_latitude_y")
        cube.coord("longitude").rename("x_longitude_x")
        sample_point = [("y_latitude_y", 88), ("x_longitude_x", 25)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(0,)])

    def test_alternate_nonlatlon_names_different(self):
        # Check that (88, 25) is **NOT** closer to (88, 0) than to (87, 25)
        # ... by plain XY euclidean-distance, if coords have non-latlon names.
        cube = self._testcube_latlon_1d(lats=[88, 87], lons=[0, 25])
        cube.coord("latitude").rename("y")
        cube.coord("longitude").rename("x")
        sample_point = [("y", 88), ("x", 25)]
        result = nn_ndinds(cube, sample_point)
        self.assertEqual(result, [(1,)])

    def test_lons_wrap_359_0(self):
        # Check that (0, 359) is closer to (0, 0) than to (0, 350)
        self._check_latlon_1d(
            lats=[0, 0],
            lons=[0, 350],
            sample_point=[("latitude", 0), ("longitude", 359)],
            expect=0,
        )

    def test_lons_wrap_359_neg1(self):
        # Check that (0, 359) is closer to (0, -1) than to (0, 350)
        self._check_latlon_1d(
            lats=[0, 0],
            lons=[350, -1],
            sample_point=[("latitude", 0), ("longitude", 359)],
            expect=1,
        )

    def test_lons_wrap_neg179_plus179(self):
        # Check that (0, -179) is closer to (0, 179) than to (0, -170)
        self._check_latlon_1d(
            lats=[0, 0],
            lons=[-170, 179],
            sample_point=[("latitude", 0), ("longitude", -179)],
            expect=1,
        )

    def test_lons_over_pole(self):
        # Check that (89, 0) is closer to (89, 180) than to (85, 0)
        self._check_latlon_1d(
            lats=[85, 89],
            lons=[0, 180],
            sample_point=[("latitude", 89), ("longitude", 0)],
            expect=1,
        )


if __name__ == "__main__":
    tests.main()
