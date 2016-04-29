# (C) British Crown Copyright 2016, Met Office
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
Tests to check the validity of replacing
"iris.analysis._interpolate.regrid`('nearest')" with
"iris.cube.Cube.regrid(scheme=iris.analysis.Nearest())".

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.analysis._interpolate_private import regrid
from iris.analysis import Nearest
from iris.cube import Cube
from iris.coords import AuxCoord, DimCoord


def grid_cube(xx, yy, data=None):
    nx, ny = len(xx), len(yy)
    if data is not None:
        data = np.array(data).reshape((ny, nx))
    else:
        data = np.zeros((ny, nx))
    cube = Cube(data)
    y_coord = DimCoord(yy, standard_name='latitude', units='degrees')
    x_coord = DimCoord(xx, standard_name='longitude', units='degrees')
    cube.add_dim_coord(y_coord, 0)
    cube.add_dim_coord(x_coord, 1)
    return cube


ENABLE_DEBUG_OUTPUT = False


def _debug_data(cube, test_id):
    if ENABLE_DEBUG_OUTPUT:
        print
        data = cube.data
        print('CUBE: {}'.format(test_id))
        print('  x={!r}'.format(cube.coord('longitude').points))
        print('  y={!r}'.format(cube.coord('latitude').points))
        print('data[{}]:'.format(type(data)))
        print(repr(data))


class MixinCheckingCode(object):
    def test_basic(self):
        src_x = [30., 40., 50.]
        dst_x = [32., 42.]
        src_y = [-10., 0., 10.]
        dst_y = [-8., 2.]
        data = [[3., 4., 5.],
                [23., 24., 25.],
                [43., 44., 45.]]
        expected_result = [[3., 4.],
                           [23., 24.]]
        src_cube = grid_cube(src_x, src_y, data)
        _debug_data(src_cube, "basic SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "basic RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_src_extrapolation(self):
        src_x = [30., 40., 50.]
        dst_x = [0., 29.0, 39.0]
        src_y = [-10., 0., 10.]
        dst_y = [-50., -9., -1.]
        data = [[3., 4., 5.],
                [23., 24., 25.],
                [43., 44., 45.]]
        expected_result = [[3., 3., 4.],
                           [3., 3., 4.],
                           [23., 23., 24.]]
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
        data = [[3., 4., 5.],
                [23., 24., 25.],
                [43., 44., 45.]]
        expected_result = [[3., 4., 4., 4., 5.],
                           [23., 24., 24., 24., 25.],
                           [23., 24., 24., 24., 25.],
                           [23., 24., 24., 24., 25.],
                           [43., 44., 44., 44., 45.]]
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
        data = np.ma.masked_equal([[3., 4., 5.],
                                   [23., 999, 25.],
                                   [43., 44., 45.]],
                                  999)
        expected_result = np.ma.masked_equal(
            [[3., 4., 4., 4., 4., 4., 5.],
             [23., 999, 999, 999, 999, 999, 25.],
             [23., 999, 999, 999, 999, 999, 25.],
             [23., 999, 999, 999, 999, 999, 25.],
             [23., 999, 999, 999, 999, 999, 25.],
             [23., 999, 999, 999, 999, 999, 25.],
             [43., 44., 44., 44., 44., 44., 45.]],
            999)
        src_cube = grid_cube(src_x, src_y, data)
        src_cube.data = np.ma.masked_array(src_cube.data)
        src_cube.data[1, 1] = np.ma.masked
        _debug_data(src_cube, "masked SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(src_cube, dst_cube,
                                  translate_nans_to_mask=True)
        _debug_data(result_cube, "masked RESULT")
        self.assertMaskedArrayEqual(result_cube.data, expected_result)

    def test_wrapping_non_circular(self):
        src_x = [-10., 0., 10.]
        dst_x = [-360.0, -170., -1.0, 1.0, 50.0, 170.0, 352.0, 720.0]
        src_y = [0., 10.]
        dst_y = [0., 10.]
        data = [[3., 4., 5.],
                [3., 4., 5.]]
        src_cube = grid_cube(src_x, src_y, data)
        dst_cube = grid_cube(dst_x, dst_y)
        # Account for a behavioural difference in this case :
        # The Nearest scheme does wrapping of modular coordinate values.
        # Thus target of 352.0 --> -8.0, which is nearest to -10.
        # This looks just like "circular" handling, but only because it happens
        # to produce the same results *for nearest-neighbour in particular*.
        if isinstance(self, TestInterpolateRegridNearest):
            # interpolate.regrid --> Wrapping-free results (non-circular).
            expected_result = [[3., 3., 4., 4., 5., 5., 5., 5.],
                               [3., 3., 4., 4., 5., 5., 5., 5.]]
        else:
            # cube regrid --> Wrapped results.
            expected_result = [[4., 3., 4., 4., 5., 5., 3., 4.],
                               [4., 3., 4., 4., 5., 5., 3., 4.]]
        _debug_data(src_cube, "noncircular SOURCE")
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "noncircular RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_wrapping_circular(self):
        # When x-coord is "circular", the above distinction does not apply :
        # results are the same for both calculations.
        src_x = [-10., 0., 10.]
        dst_x = [-360.0, -170., -1.0, 1.0, 50.0, 170.0, 352.0, 720.0]
        src_y = [0., 10.]
        dst_y = [0., 10.]
        data = [[3., 4., 5.],
                [3., 4., 5.]]
        src_cube = grid_cube(src_x, src_y, data)
        dst_cube = grid_cube(dst_x, dst_y)
        src_cube.coord('longitude').circular = True
        expected_result = [[4., 3., 4., 4., 5., 5., 3., 4.],
                           [4., 3., 4., 4., 5., 5., 3., 4.]]
        _debug_data(src_cube, "circular SOURCE")
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "circular RESULT")
        self.assertArrayAllClose(result_cube.data, expected_result)

    def test_wrapping_non_angular(self):
        src_x = [-10., 0., 10.]
        dst_x = [-360.0, -170., -1.0, 1.0, 50.0, 170.0, 352.0, 720.0]
        src_y = [0., 10.]
        dst_y = [0., 10.]
        data = [[3., 4., 5.],
                [3., 4., 5.]]
        src_cube = grid_cube(src_x, src_y, data)
        dst_cube = grid_cube(dst_x, dst_y)
        for co_name in ('longitude', 'latitude'):
            for cube in (src_cube, dst_cube):
                coord = cube.coord(co_name)
                coord.coord_system = None
                coord.convert_units('1')
        # interpolate.regrid --> Wrapping-free results (non-circular).
        expected_result = [[3., 3., 4., 4., 5., 5., 5., 5.],
                           [3., 3., 4., 4., 5., 5., 5., 5.]]
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
        data = [[3., 4., 5.],
                [23., nan, 25.],
                [43., 44., 45.]]
        expected_result = [[3., 4., 4., 4., 4., 4., 5.],
                           [23., nan, nan, nan, nan, nan, 25.],
                           [23., nan, nan, nan, nan, nan, 25.],
                           [23., nan, nan, nan, nan, nan, 25.],
                           [23., nan, nan, nan, nan, nan, 25.],
                           [23., nan, nan, nan, nan, nan, 25.],
                           [43., 44., 44., 44., 44., 44., 45.]]
        src_cube = grid_cube(src_x, src_y, data)
        _debug_data(src_cube, "nan SOURCE")
        dst_cube = grid_cube(dst_x, dst_y)
        result_cube = self.regrid(src_cube, dst_cube)
        _debug_data(result_cube, "nan RESULT")
        self.assertArrayEqual(result_cube.data, expected_result)


# perform identical tests on the old + new approaches
class TestInterpolateRegridNearest(MixinCheckingCode, tests.IrisTest):
    def regrid(self, src_cube, dst_cube,
               translate_nans_to_mask=False, **kwargs):
        result = regrid(src_cube, dst_cube, mode='nearest')
        data = result.data
        if translate_nans_to_mask and np.any(np.isnan(data)):
            data = np.ma.masked_array(data, mask=np.isnan(data))
            result.data = data
        return result


class TestCubeRegridNearest(MixinCheckingCode, tests.IrisTest):
    scheme = Nearest(extrapolation_mode='extrapolate')

    def regrid(self, src_cube, dst_cube, **kwargs):
        return src_cube.regrid(dst_cube, scheme=self.scheme)


if __name__ == '__main__':
    tests.main()
