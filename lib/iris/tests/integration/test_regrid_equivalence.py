# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Tests to check the validity of replacing
"iris.analysis._interpolate.regrid`('nearest')" with
"iris.cube.Cube.regrid(scheme=iris.analysis.Nearest())".

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.analysis import Nearest
from iris.coords import DimCoord
from iris.cube import Cube


def grid_cube(xx, yy, data=None):
    nx, ny = len(xx), len(yy)
    if data is not None:
        data = np.array(data).reshape((ny, nx))
    else:
        data = np.zeros((ny, nx))
    cube = Cube(data)
    y_coord = DimCoord(yy, standard_name="latitude", units="degrees")
    x_coord = DimCoord(xx, standard_name="longitude", units="degrees")
    cube.add_dim_coord(y_coord, 0)
    cube.add_dim_coord(x_coord, 1)
    return cube


ENABLE_DEBUG_OUTPUT = False


def _debug_data(cube, test_id):
    if ENABLE_DEBUG_OUTPUT:
        print
        data = cube.data
        print("CUBE: {}".format(test_id))
        print("  x={!r}".format(cube.coord("longitude").points))
        print("  y={!r}".format(cube.coord("latitude").points))
        print("data[{}]:".format(type(data)))
        print(repr(data))


class MixinCheckingCode:
    def test_basic(self):
        src_x = [30.0, 40.0, 50.0]
        dst_x = [32.0, 42.0]
        src_y = [-10.0, 0.0, 10.0]
        dst_y = [-8.0, 2.0]
        data = [[3.0, 4.0, 5.0], [23.0, 24.0, 25.0], [43.0, 44.0, 45.0]]
        expected_result = [[3.0, 4.0], [23.0, 24.0]]
        src_cube = grid_cube(src_x, src_y, data)
        _debug_data(src_cube, "basic SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "basic RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_src_extrapolation(self):
        src_x = [30.0, 40.0, 50.0]
        dst_x = [0.0, 29.0, 39.0]
        src_y = [-10.0, 0.0, 10.0]
        dst_y = [-50.0, -9.0, -1.0]
        data = [[3.0, 4.0, 5.0], [23.0, 24.0, 25.0], [43.0, 44.0, 45.0]]
        expected_result = [
            [3.0, 3.0, 4.0],
            [3.0, 3.0, 4.0],
            [23.0, 23.0, 24.0],
        ]
        src_cube = grid_cube(src_x, src_y, data)
        _debug_data(src_cube, "extrapolate SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "extrapolate RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_exact_matching_points(self):
        src_x = [10.0, 20.0, 30.0]
        src_y = [10.0, 20.0, 30.0]
        dst_x = [14.9, 15.1, 20.0, 24.9, 25.1]
        dst_y = [14.9, 15.1, 20.0, 24.9, 25.1]
        data = [[3.0, 4.0, 5.0], [23.0, 24.0, 25.0], [43.0, 44.0, 45.0]]
        expected_result = [
            [3.0, 4.0, 4.0, 4.0, 5.0],
            [23.0, 24.0, 24.0, 24.0, 25.0],
            [23.0, 24.0, 24.0, 24.0, 25.0],
            [23.0, 24.0, 24.0, 24.0, 25.0],
            [43.0, 44.0, 44.0, 44.0, 45.0],
        ]
        src_cube = grid_cube(src_x, src_y, data)
        _debug_data(src_cube, "matching SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "matching RESULt")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_source_mask(self):
        src_x = [40.0, 50.0, 60.0]
        src_y = [40.0, 50.0, 60.0]
        dst_x = [44.99, 45.01, 48.0, 50.0, 52.0, 54.99, 55.01]
        dst_y = [44.99, 45.01, 48.0, 50.0, 52.0, 54.99, 55.01]
        data = np.ma.masked_equal(
            [[3.0, 4.0, 5.0], [23.0, 999, 25.0], [43.0, 44.0, 45.0]], 999
        )
        expected_result = np.ma.masked_equal(
            [
                [3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0],
                [23.0, 999, 999, 999, 999, 999, 25.0],
                [23.0, 999, 999, 999, 999, 999, 25.0],
                [23.0, 999, 999, 999, 999, 999, 25.0],
                [23.0, 999, 999, 999, 999, 999, 25.0],
                [23.0, 999, 999, 999, 999, 999, 25.0],
                [43.0, 44.0, 44.0, 44.0, 44.0, 44.0, 45.0],
            ],
            999,
        )
        src_cube = grid_cube(src_x, src_y, data)
        src_cube.data = np.ma.masked_array(src_cube.data)
        src_cube.data[1, 1] = np.ma.masked
        _debug_data(src_cube, "masked SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(
            src_cube, dst_cube, translate_nans_to_mask=True
        )
        _debug_data(result_cube, "masked RESULT")
        self.assertMaskedArrayEqual(result_cube.data, expected_result)

    def test_wrapping_non_circular(self):
        src_x = [-10.0, 0.0, 10.0]
        dst_x = [-360.0, -170.0, -1.0, 1.0, 50.0, 170.0, 352.0, 720.0]
        src_y = [0.0, 10.0]
        dst_y = [0.0, 10.0]
        data = [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]
        src_cube = grid_cube(src_x, src_y, data)
        dst_cube = grid_cube(dst_x, dst_y)
        expected_result = [
            [4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 4.0],
            [4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 4.0],
        ]
        _debug_data(src_cube, "noncircular SOURCE")
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "noncircular RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_wrapping_circular(self):
        # When x-coord is "circular", the above distinction does not apply :
        # results are the same for both calculations.
        src_x = [-10.0, 0.0, 10.0]
        dst_x = [-360.0, -170.0, -1.0, 1.0, 50.0, 170.0, 352.0, 720.0]
        src_y = [0.0, 10.0]
        dst_y = [0.0, 10.0]
        data = [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]
        src_cube = grid_cube(src_x, src_y, data)
        dst_cube = grid_cube(dst_x, dst_y)
        src_cube.coord("longitude").circular = True
        expected_result = [
            [4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 4.0],
            [4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 4.0],
        ]
        _debug_data(src_cube, "circular SOURCE")
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "circular RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_wrapping_non_angular(self):
        src_x = [-10.0, 0.0, 10.0]
        dst_x = [-360.0, -170.0, -1.0, 1.0, 50.0, 170.0, 352.0, 720.0]
        src_y = [0.0, 10.0]
        dst_y = [0.0, 10.0]
        data = [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]
        src_cube = grid_cube(src_x, src_y, data)
        dst_cube = grid_cube(dst_x, dst_y)
        for co_name in ("longitude", "latitude"):
            for cube in (src_cube, dst_cube):
                coord = cube.coord(co_name)
                coord.coord_system = None
                coord.convert_units("1")
        # interpolate.regrid --> Wrapping-free results (non-circular).
        expected_result = [
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0],
        ]
        _debug_data(src_cube, "non-angle-lons SOURCE")
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "non-angle-lons RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_source_nan(self):
        src_x = [40.0, 50.0, 60.0]
        src_y = [40.0, 50.0, 60.0]
        dst_x = [44.99, 45.01, 48.0, 50.0, 52.0, 54.99, 55.01]
        dst_y = [44.99, 45.01, 48.0, 50.0, 52.0, 54.99, 55.01]
        nan = np.nan
        data = [[3.0, 4.0, 5.0], [23.0, nan, 25.0], [43.0, 44.0, 45.0]]
        expected_result = [
            [3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0],
            [23.0, nan, nan, nan, nan, nan, 25.0],
            [23.0, nan, nan, nan, nan, nan, 25.0],
            [23.0, nan, nan, nan, nan, nan, 25.0],
            [23.0, nan, nan, nan, nan, nan, 25.0],
            [23.0, nan, nan, nan, nan, nan, 25.0],
            [43.0, 44.0, 44.0, 44.0, 44.0, 44.0, 45.0],
        ]
        src_cube = grid_cube(src_x, src_y, data)
        _debug_data(src_cube, "nan SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "nan RESULT")
        self.assertArrayEqual(result_cube.data, expected_result)


class TestCubeRegridNearest(MixinCheckingCode, tests.IrisTest):
    scheme = Nearest(extrapolation_mode="extrapolate")

    def regrid(self, src_cube, dst_cube, **kwargs):
        return src_cube.regrid(dst_cube, scheme=self.scheme)


if __name__ == "__main__":
    tests.main()
