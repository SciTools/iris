# (C) British Crown Copyright 2010 - 2016, Met Office
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
Test the interpolation of Iris cubes.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d

import iris
import iris.coord_systems
import iris.cube
import iris.analysis.interpolate
import iris.tests.stock
import iris.analysis._interpolate_private as iintrp


def normalise_order(cube):
    # Avoid the crazy array ordering which results from using:
    #   * np.append() in NumPy 1.6, which is triggered in the `linear()`
    #     function when the circular flag is true.
    #   * scipy.interpolate.interp1d in 0.11.0 which is used in
    #     `Linear1dExtrapolator`.
    cube.data = np.ascontiguousarray(cube.data)


class TestLinearExtrapolator(tests.IrisTest):
    def test_simple_axis0(self):
        a = np.arange(12.).reshape(3, 4)
        r = iintrp.Linear1dExtrapolator(interp1d(np.arange(3), a, axis=0))
        
        np.testing.assert_array_equal(r(0), np.array([ 0.,  1.,  2.,  3.]))
        np.testing.assert_array_equal(r(-1), np.array([-4.,  -3.,  -2., -1.]))
        np.testing.assert_array_equal(r(3), np.array([ 12.,  13.,  14.,  15.]))        
        np.testing.assert_array_equal(r(2.5), np.array([ 10.,  11.,  12.,  13.]))
        
        # 2 Non-extrapolation point
        np.testing.assert_array_equal(r(np.array([1.5, 2])), np.array([[  6.,   7.,   8.,   9.],
                                                                       [  8.,   9.,  10.,  11.]]))
        
        # 1 Non-extrapolation point & 1 upper value extrapolation
        np.testing.assert_array_equal(r(np.array([1.5, 3])), np.array([[  6.,   7.,   8.,   9.],
                                                                       [ 12.,  13.,  14.,  15.]]))
        
        # 2 upper value extrapolation
        np.testing.assert_array_equal(r(np.array([2.5, 3])), np.array([[ 10.,  11.,  12.,  13.],
                                                                       [ 12.,  13.,  14.,  15.]]))

        # 1 lower value extrapolation & 1 Non-extrapolation point
        np.testing.assert_array_equal(r(np.array([-1, 1.5])), np.array([[-4., -3., -2., -1.],
                                                                        [ 6.,  7.,  8.,  9.]]))

        # 2 lower value extrapolation
        np.testing.assert_array_equal(r(np.array([-1.5, -1])), np.array([[-6., -5., -4., -3.],
                                                                         [-4., -3., -2., -1.]]))
        
        # 2 lower value extrapolation, 2 Non-extrapolation point & 2 upper value extrapolation
        np.testing.assert_array_equal(r(np.array([-1.5, -1, 1, 1.5, 2.5, 3])),
                                         np.array([[ -6.,  -5.,  -4.,  -3.],
                                                   [ -4.,  -3.,  -2.,  -1.],
                                                   [  4.,   5.,   6.,   7.],
                                                   [  6.,   7.,   8.,   9.],
                                                   [ 10.,  11.,  12.,  13.],
                                                   [ 12.,  13.,  14.,  15.]]))
   
    def test_simple_axis1(self):
        a = np.arange(12).reshape(3, 4)
        r = iintrp.Linear1dExtrapolator(interp1d(np.arange(4), a, axis=1))
        
        # check non-extrapolation given the Extrapolator object 
        np.testing.assert_array_equal(r(0), np.array([ 0.,  4.,  8.]))
        
        # check the result's shape in a 1d array (of len 0 & 1)
        np.testing.assert_array_equal(r(np.array(0)), np.array([ 0.,  4.,  8.]))        
        np.testing.assert_array_equal(r(np.array([0])), np.array([ [0.],  [4.],  [8.]]))
        
        # check extrapolation below the minimum value (and check the equivalent 0d & 1d arrays)
        np.testing.assert_array_equal(r(-1), np.array([-1., 3., 7.]))
        np.testing.assert_array_equal(r(np.array(-1)), np.array([-1., 3., 7.]))
        np.testing.assert_array_equal(r(np.array([-1])), np.array([[-1.], [ 3.], [ 7.]]))
        
        # check extrapolation above the maximum value
        np.testing.assert_array_equal(r(3), np.array([  3.,   7.,  11.]))
        np.testing.assert_array_equal(r(2.5), np.array([  2.5,   6.5,  10.5]))
        
        # 2 Non-extrapolation point 
        np.testing.assert_array_equal(r(np.array([1.5, 2])), np.array([[  1.5,   2. ],
                                                                       [  5.5,   6. ],
                                                                       [  9.5,  10. ]]))
        
        # 1 Non-extrapolation point & 1 upper value extrapolation
        np.testing.assert_array_equal(r(np.array([1.5, 5])), np.array([[  1.5,   5. ],
                                                                       [  5.5,   9. ],
                                                                       [  9.5,  13. ]]))
        
        # 2 upper value extrapolation        
        np.testing.assert_array_equal(r(np.array([4.5, 5])), np.array([[  4.5,   5. ],
                                                                       [  8.5,   9. ],
                                                                       [ 12.5,  13. ]]))

        # 1 lower value extrapolation & 1 Non-extrapolation point
        np.testing.assert_array_equal(r(np.array([-0.5, 1.5])), np.array([[-0.5,  1.5],
                                                                          [ 3.5,  5.5],
                                                                          [ 7.5,  9.5]]))
        
        # 2 lower value extrapolation
        np.testing.assert_array_equal(r(np.array([-1.5, -1])), np.array([[-1.5, -1. ],
                                                                         [ 2.5,  3. ],
                                                                         [ 6.5,  7. ]]))
        
        # 2 lower value extrapolation, 2 Non-extrapolation point & 2 upper value extrapolation
        np.testing.assert_array_equal(r(np.array([-1.5, -1, 1.5, 2, 4.5, 5])), 
                                      np.array([[ -1.5, -1., 1.5,  2.,  4.5,  5. ],
                                                [  2.5,  3., 5.5,  6.,  8.5,  9. ],
                                                [  6.5,  7., 9.5, 10., 12.5, 13. ]]))
        
        
    def test_simple_3d_axis1(self):
        a = np.arange(24.).reshape(3, 4, 2)
        r = iintrp.Linear1dExtrapolator(interp1d(np.arange(4.), a, axis=1))
        
#       a:
#        [[[ 0  1]
#          [ 2  3]
#          [ 4  5]
#          [ 6  7]]
#        
#         [[ 8  9]
#          [10 11]
#          [12 13]
#          [14 15]]
#        
#         [[16 17]
#          [18 19]
#          [20 21]
#          [22 23]]
#         ] 

        np.testing.assert_array_equal(r(0), np.array([[  0.,   1.],
                                                      [  8.,   9.],
                                                      [ 16.,  17.]]))
        
        np.testing.assert_array_equal(r(1), np.array([[  2.,   3.],
                                                      [ 10.,  11.],
                                                      [ 18.,  19.]]))
        
        np.testing.assert_array_equal(r(-1), np.array([[ -2.,  -1.],
                                                       [  6.,   7.],
                                                       [ 14.,  15.]]))
        
        np.testing.assert_array_equal(r(4), np.array([[  8.,   9.],
                                                      [ 16.,  17.],
                                                      [ 24.,  25.]]))

        np.testing.assert_array_equal(r(0.25), np.array([[  0.5,   1.5],
                                                         [  8.5,   9.5],
                                                         [ 16.5,  17.5]]))
        
        np.testing.assert_array_equal(r(-0.25), np.array([[ -0.5,   0.5],
                                                          [  7.5,   8.5],
                                                          [ 15.5,  16.5]]))
        
        np.testing.assert_array_equal(r(4.25), np.array([[  8.5,   9.5],
                                                         [ 16.5,  17.5],
                                                         [ 24.5,  25.5]]))
        
        np.testing.assert_array_equal(r(np.array([0.5, 1])), np.array([[[  1.,   2.], [  2.,   3.]],
                                                                       [[  9.,  10.], [ 10.,  11.]],
                                                                       [[ 17.,  18.], [ 18.,  19.]]]))
        
        np.testing.assert_array_equal(r(np.array([0.5, 4])), np.array([[[  1.,   2.], [  8.,   9.]],
                                                                       [[  9.,  10.], [ 16.,  17.]],
                                                                       [[ 17.,  18.], [ 24.,  25.]]]))

        np.testing.assert_array_equal(r(np.array([-0.5, 0.5])), np.array([[[ -1.,   0.], [  1.,   2.]],
                                                                          [[  7.,   8.], [  9.,  10.]],
                                                                          [[ 15.,  16.], [ 17.,  18.]]]))

        np.testing.assert_array_equal(r(np.array([-1.5, -1, 0.5, 1, 4.5, 5])), 
                                         np.array([[[ -3.,  -2.], [ -2.,  -1.], [  1.,   2.], [  2.,   3.], [  9.,  10.], [ 10.,  11.]],
                                                   [[  5.,   6.], [  6.,   7.], [  9.,  10.], [ 10.,  11.], [ 17.,  18.], [ 18.,  19.]],
                                                   [[ 13.,  14.], [ 14.,  15.], [ 17.,  18.], [ 18.,  19.], [ 25.,  26.], [ 26.,  27.]]]))
        
    def test_variable_gradient(self):
        a = np.array([[2, 4, 8], [0, 5, 11]])
        r = iintrp.Linear1dExtrapolator(interp1d(np.arange(2), a, axis=0))
        
        np.testing.assert_array_equal(r(0), np.array([ 2.,  4.,  8.]))
        np.testing.assert_array_equal(r(-1), np.array([ 4.,  3.,  5.]))
        np.testing.assert_array_equal(r(3), np.array([ -4.,   7.,  17.]))        
        np.testing.assert_array_equal(r(2.5), np.array([ -3.,   6.5,  15.5]))
        
        np.testing.assert_array_equal(r(np.array([1.5, 2])), np.array([[ -1.,   5.5,  12.5],
                                                                       [ -2.,   6.,  14. ]]))
    
        np.testing.assert_array_equal(r(np.array([-1.5, 3.5])), np.array([[  5.,   2.5,   3.5],
                                                                          [ -5.,   7.5,  18.5]]))


class TestLinearLengthOneCoord(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.lat_lon_cube()
        self.cube.data = self.cube.data.astype(float)

    def test_single_point(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        r = iintrp.linear(cube, [('longitude', [1.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_single_pt_0'))

        # Slice to form (1, 4) shaped cube.
        cube = self.cube[1:2, :]
        r = iintrp.linear(cube, [('latitude', [1.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_single_pt_1'))

    def test_multiple_points(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        r = iintrp.linear(cube, [('longitude',
                                                     [1., 2., 3., 4.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_0'))

        # Slice to form (1, 4) shaped cube.
        cube = self.cube[1:2, :]
        r = iintrp.linear(cube, [('latitude',
                                                     [1., 2., 3., 4.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_1'))

    def test_single_point_to_scalar(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        r = iintrp.linear(cube, [('longitude', 1.)])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_scalar_0'))

        # Slice to form (1, 4) shaped cube.
        cube = self.cube[1:2, :]
        r = iintrp.linear(cube, [('latitude', 1.)])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_scalar_1'))

    def test_extrapolation_mode_same_pt(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        src_points = cube.coord('longitude').points
        r = iintrp.linear(cube, [('longitude', src_points)],
                                             extrapolation_mode='linear')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_same_pt'))
        r = iintrp.linear(cube, [('longitude', src_points)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_same_pt'))
        r = iintrp.linear(cube, [('longitude', src_points)],
                                             extrapolation_mode='error')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_same_pt'))

    def test_extrapolation_mode_multiple_same_pts(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        src_points = cube.coord('longitude').points
        new_points = [src_points[0]] * 3
        r = iintrp.linear(cube, [('longitude', new_points)],
                                             extrapolation_mode='linear')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_same'))
        r = iintrp.linear(cube, [('longitude', new_points)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_same'))
        r = iintrp.linear(cube, [('longitude', new_points)],
                                             extrapolation_mode='error')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_same'))

    def test_extrapolation_mode_different_pts(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        src_points = cube.coord('longitude').points
        new_points_single = src_points + 0.2
        new_points_multiple = [src_points[0],
                               src_points[0] + 0.2,
                               src_points[0] + 0.4]
        new_points_scalar = src_points[0] + 0.2

        # 'nan' mode
        r = iintrp.linear(cube, [('longitude',
                                                     new_points_single)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_single_pt_nan'))
        r = iintrp.linear(cube, [('longitude',
                                                     new_points_multiple)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_nan'))
        r = iintrp.linear(cube, [('longitude',
                                                     new_points_scalar)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_scalar_nan'))

        # 'error' mode
        with self.assertRaises(ValueError):
            r = iintrp.linear(cube, [('longitude',
                                                         new_points_single)],
                                                 extrapolation_mode='error')
        with self.assertRaises(ValueError):
            r = iintrp.linear(cube, [('longitude',
                                                         new_points_multiple)],
                                                 extrapolation_mode='error')
        with self.assertRaises(ValueError):
            r = iintrp.linear(cube, [('longitude',
                                                         new_points_scalar)],
                                                 extrapolation_mode='error')


class TestLinear1dInterpolation(tests.IrisTest):
    def setUp(self):
        data = np.arange(12., dtype=np.float32).reshape((4, 3))
        c2 = iris.cube.Cube(data)
           
        c2.long_name = 'test 2d dimensional cube'
        c2.units = 'kelvin'
        
        pts = 3 + np.arange(4, dtype=np.float32) * 2
        b = iris.coords.DimCoord(pts, long_name='dim1', units=1)
        d = iris.coords.AuxCoord([3, 3.5, 6], long_name='dim2', units=1)
        e = iris.coords.AuxCoord(3.0, long_name='an_other', units=1)

        c2.add_dim_coord(b, 0)
        c2.add_aux_coord(d, 1)
        c2.add_aux_coord(e)

        self.simple2d_cube = c2
        
        d = iris.coords.AuxCoord([5, 9, 20], long_name='shared_x_coord', units=1)
        c3 = c2.copy()
        c3.add_aux_coord(d, 1)
        self.simple2d_cube_extended = c3

        pts = 0.1 + np.arange(5, dtype=np.float32) * 0.1
        f = iris.coords.DimCoord(pts, long_name='r', units=1)
        g = iris.coords.DimCoord([0.0, 90.0, 180.0, 270.0], long_name='theta', units='degrees', circular=True)
        data = np.arange(20., dtype=np.float32).reshape((5, 4))
        c4 = iris.cube.Cube(data)
        c4.add_dim_coord(f, 0)
        c4.add_dim_coord(g, 1)
        self.simple2d_cube_circular = c4

    def test_dim_to_aux(self):
        cube = self.simple2d_cube
        other = iris.coords.DimCoord([1, 2, 3, 4], long_name='was_dim')
        cube.add_aux_coord(other, 0)
        r = iintrp.linear(cube, [('dim1', [7, 3, 5])])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'dim_to_aux.cml'))

    def test_bad_sample_point_format(self):
        self.assertRaises(TypeError, iintrp.linear, self.simple2d_cube, ('dim1', 4))
    
    def test_simple_single_point(self):
        r = iintrp.linear(self.simple2d_cube, [('dim1', 4)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_single_point.cml'), checksum=False)
        np.testing.assert_array_equal(r.data, np.array([1.5, 2.5, 3.5], dtype=self.simple2d_cube.data.dtype))
        
    def test_monotonic_decreasing_coord(self):
        c = self.simple2d_cube[::-1]
        r = iintrp.linear(c, [('dim1', 4)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_single_point.cml'), checksum=False)
        np.testing.assert_array_equal(r.data, np.array([1.5, 2.5, 3.5], dtype=self.simple2d_cube.data.dtype))
        
    def test_overspecified(self):
        self.assertRaises(ValueError, iintrp.linear, self.simple2d_cube[0, :], [('dim1', 4)])
        
    def test_bounded_coordinate(self):
        # The results should be exactly the same as for the
        # non-bounded case.
        cube = self.simple2d_cube
        cube.coord('dim1').guess_bounds()
        r = iintrp.linear(cube, [('dim1', [4, 5])])
        np.testing.assert_array_equal(r.data, np.array([[ 1.5,  2.5,  3.5], [ 3.,  4.,  5. ]]))
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))

    def test_simple_multiple_point(self):
        r = iintrp.linear(self.simple2d_cube, [('dim1', [4, 5])])
        np.testing.assert_array_equal(r.data, np.array([[ 1.5,  2.5,  3.5], [ 3.,  4.,  5. ]]))
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))
        
        # Check that numpy arrays specifications work
        r = iintrp.linear(self.simple2d_cube, [('dim1', np.array([4, 5]))])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))

    def test_circular_vs_non_circular_coord(self):
        cube = self.simple2d_cube_circular
        other = iris.coords.AuxCoord([10, 6, 7, 4], long_name='other')
        cube.add_aux_coord(other, 1)
        samples = [0, 60, 300]
        r = iintrp.linear(cube, [('theta', samples)])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'circular_vs_non_circular.cml'))

    def test_simple_multiple_points_circular(self):
        r = iintrp.linear(self.simple2d_cube_circular, [('theta', [0., 60., 120., 180.])])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points_circular.cml'))
        
        # check that the values returned by theta 0 & 360 are the same...
        r1 = iintrp.linear(self.simple2d_cube_circular, [('theta', 360)])
        r2 = iintrp.linear(self.simple2d_cube_circular, [('theta', 0)])
        np.testing.assert_array_almost_equal(r1.data, r2.data)
        
    def test_simple_multiple_coords(self):
        expected_result = np.array(2.5)
        r = iintrp.linear(self.simple2d_cube, [('dim1', 4), ('dim2', 3.5), ])
        np.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
        
        # Check that it doesn't matter if you do the interpolation in separate steps...
        r = iintrp.linear(self.simple2d_cube, [('dim2', 3.5)])
        r = iintrp.linear(r, [('dim1', 4)])
        np.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
        
        r = iintrp.linear(self.simple2d_cube, [('dim1', 4)])
        r = iintrp.linear(r, [('dim2', 3.5)])
        np.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
    
    def test_coord_not_found(self):
        self.assertRaises(KeyError, iintrp.linear, self.simple2d_cube, 
                          [('non_existant_coord', [3.5, 3.25])])
    
    def test_simple_coord_error_extrapolation(self):
        self.assertRaises(ValueError, iintrp.linear, self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='error')

    def test_simple_coord_linear_extrapolation(self):
        r = iintrp.linear( self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='linear')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_linear_extrapolation.cml'))
        
        np.testing.assert_array_equal(r.data, np.array([-1.,  2.,  5.,  8.], dtype=self.simple2d_cube.data.dtype))
        
        r = iintrp.linear(self.simple2d_cube, [('dim1', 1)])
        np.testing.assert_array_equal(r.data, np.array([-3., -2., -1.], dtype=self.simple2d_cube.data.dtype))
                
    def test_simple_coord_linear_extrapolation_multipoint1(self):
        r = iintrp.linear( self.simple2d_cube, [('dim1', [-1, 1, 10, 11])], extrapolation_mode='linear')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_linear_extrapolation_multipoint1.cml'))
        
    def test_simple_coord_linear_extrapolation_multipoint2(self):
        r = iintrp.linear( self.simple2d_cube, [('dim1', [1, 10])], extrapolation_mode='linear')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_linear_extrapolation_multipoint2.cml'))
        
    def test_simple_coord_nan_extrapolation(self):
        r = iintrp.linear( self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='nan')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_nan_extrapolation.cml'))
    
    def test_multiple_coord_extrapolation(self):
        self.assertRaises(ValueError, iintrp.linear, self.simple2d_cube, [('dim2', 2.5), ('dim1', 12.5)], extrapolation_mode='error')    
        
    def test_multiple_coord_linear_extrapolation(self):
        r = iintrp.linear(self.simple2d_cube, [('dim2', 9), ('dim1', 1.5)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords_extrapolation.cml'))
        
    def test_lots_of_points(self):
        r = iintrp.linear(self.simple2d_cube, [('dim1', np.linspace(3, 9, 20))])
        # XXX Implement a test!?!
        
    def test_shared_axis(self):
        c = self.simple2d_cube_extended
        r = iintrp.linear(c, [('dim2', [3.5, 3.25])])
        normalise_order(r)
        
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_shared_axis.cml'))
        
        self.assertRaises(ValueError, iintrp.linear, c, [('dim2', [3.5, 3.25]), ('shared_x_coord', [9, 7])])

    def test_points_datatype_casting(self):
        # this test tries to extract a float from an array of type integer. the result should be of type float.
        r = iintrp.linear(self.simple2d_cube_extended, [('shared_x_coord', 7.5)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_casting_datatype.cml'))


@tests.skip_data
class TestNearestLinearInterpolRealData(tests.IrisTest):
    def setUp(self):
        file = tests.get_data_path(('PP', 'globClim1', 'theta.pp'))
        self.cube = iris.load_cube(file)

    def test_slice(self):
        r = iintrp.linear(self.cube, [('latitude', 0)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'real_2dslice.cml'))
    
    def test_2slices(self):
        r = iintrp.linear(self.cube, [('latitude', 0.0), ('longitude', 0.0)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'real_2slices.cml'))

    def test_circular(self):
        res = iintrp.linear(self.cube,
                                               [('longitude', 359.8)])
        normalise_order(res)
        lon_coord = self.cube.coord('longitude').points
        expected = self.cube.data[..., 0] + \
            ((self.cube.data[..., -1] - self.cube.data[..., 0]) *
             (((360 - 359.8) - lon_coord[0]) /
              ((360 - lon_coord[-1]) - lon_coord[0])))
        self.assertArrayAllClose(res.data, expected, rtol=1.0e-6)

        # check that the values returned by lon 0 & 360 are the same...
        r1 = iintrp.linear(self.cube, [('longitude', 360)])
        r2 = iintrp.linear(self.cube, [('longitude', 0)])
        np.testing.assert_array_equal(r1.data, r2.data)

        self.assertCML(res, ('analysis', 'interpolation', 'linear',
                             'real_circular_2dslice.cml'), checksum=False)


class MixinNearestNeighbour(object):
    # Define standard tests for the three 'nearest_neighbour' routines.
    # Cast as a 'mixin' as it used to test both (a) the original routines and
    # (b) replacement operations to justify deprecation.

    def _common_setUp(self):
        self.cube = iris.tests.stock.global_pp()
        points = np.arange(self.cube.coord('latitude').shape[0], dtype=np.float32)
        coord_to_add = iris.coords.DimCoord(points, long_name='i', units='meters')
        self.cube.add_aux_coord(coord_to_add, 0)

    def test_nearest_neighbour(self):
        point_spec = [('latitude', 40), ('longitude', 39)]
        
        indices = iintrp.nearest_neighbour_indices(self.cube, point_spec)
        self.assertEqual(indices, (20, 10))
        
        b = iintrp.extract_nearest_neighbour(self.cube, point_spec) 

        # Check that the data has not been loaded on either the original cube,
        # nor the interpolated one.
        self.assertTrue(b.has_lazy_data())
        self.assertTrue(self.cube.has_lazy_data())
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude_longitude.cml'))
        
        value = iintrp.nearest_neighbour_data_value(self.cube, point_spec)
        self.assertEqual(value, np.array(285.98785, dtype=np.float32))

        # Check that the value back is that which was returned by the extract method
        self.assertEqual(value, b.data)
        
    def test_nearest_neighbour_slice(self):
        point_spec = [('latitude', 40)]
        indices = iintrp.nearest_neighbour_indices(self.cube, point_spec)
        self.assertEqual(indices, (20, slice(None, None)))

        b = iintrp.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude.cml'))
        
        # cannot get a specific point from these point specifications
        self.assertRaises(ValueError, iintrp.nearest_neighbour_data_value, self.cube, point_spec)

    def test_nearest_neighbour_over_specification_which_is_consistent(self):
        # latitude 40 is the 20th point
        point_spec = [('latitude', 40), ('i', 20), ('longitude', 38)]
        
        indices = iintrp.nearest_neighbour_indices(self.cube, point_spec)
        self.assertEqual(indices, (20, 10))
        
        b = iintrp.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude_longitude.cml'))
        
        value = iintrp.nearest_neighbour_data_value(self.cube, point_spec)
        # Check that the value back is that which was returned by the extract method
        self.assertEqual(value, b.data)

    def test_nearest_neighbour_over_specification_mis_aligned(self):
        # latitude 40 is the 20th point
        point_spec = [('latitude', 40), ('i', 10), ('longitude', 38)]
        
        # assert that we get a ValueError for over specifying our interpolation
        self.assertRaises(ValueError, iintrp.nearest_neighbour_data_value, self.cube, point_spec)

    def test_nearest_neighbour_bounded_simple(self):
        point_spec = [('latitude', 37), ('longitude', 38)]
    
        coord = self.cube.coord('latitude')
        coord.guess_bounds(0.5)
    
        b = iintrp.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_bounded.cml'))
        
    def test_nearest_neighbour_bounded_requested_midpoint(self):
        # This test checks the "point inside cell" logic
        point_spec = [('latitude', 38), ('longitude', 38)]
    
        coord = self.cube.coord('latitude')
        coord.guess_bounds(0.5)
        
        b = iintrp.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_bounded_mid_point.cml'))
    
    def test_nearest_neighbour_locator_style_coord(self):
        point_spec = [('latitude', 39)]
        
        b = iintrp.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude.cml'))

    def test_nearest_neighbour_circular(self):
        # test on non-circular coordinate (latitude)
        lat_vals = np.array([
            [-150.0, -90], [-97, -90], [-92, -90], [-91, -90],  [-90.1, -90],
            [-90.0, -90],  [-89.9, -90],
            [-89, -90],  [-88, -87.5],  [-87, -87.5],
            [-86, -85], [-85.5, -85],
            [81, 80], [84, 85], [84.8, 85], [85, 85], [86, 85],
            [87, 87.5], [88, 87.5], [89, 90],
            [89.9, 90], [90.0,  90], [90.1, 90],
            [95, 90], [100, 90], [150, 90]])
        lat_test_vals = lat_vals[:, 0]
        lat_expect_vals = lat_vals[:, 1]
        lat_coord_vals = self.cube.coord('latitude').points

        def near_value(val, vals):
            # return the *exact* value from vals that is closest to val.
            # - and raise an exception if there isn't a close match.
            best_val = vals[np.argmin(np.abs(vals - val))]
            if val == 0.0:
                # absolute tolerance to 0.0 (ok for magnitudes >= 1.0 or so)
                error_level = best_val
            else:
                # calculate relative-tolerance
                error_level = abs(0.5 * (val - best_val) / (val + best_val))
            self.assertTrue(error_level < 1.0e-6,
                            'error_level {}% match of {} to one of {}'.format(
                                100.0 * error_level, val, vals))
            return best_val

        lat_expect_vals = [near_value(v, lat_coord_vals)
                           for v in lat_expect_vals]
        lat_nearest_inds = [
            iintrp.nearest_neighbour_indices(
                self.cube, [('latitude', point_val)])
            for point_val in lat_test_vals]
        lat_nearest_vals = [lat_coord_vals[i[0]] for i in lat_nearest_inds]
        self.assertArrayAlmostEqual(lat_nearest_vals, lat_expect_vals)

        # repeat with *circular* coordinate (longitude)
        lon_vals = np.array([
            [0.0, 0.0],
            [-3.75, 356.25],
            [-1.0, 0], [-0.01, 0], [0.5, 0],
            [2, 3.75], [3, 3.75], [4, 3.75], [5, 3.75], [6, 7.5],
            [350.5, 348.75], [351, 352.5], [354, 352.5],
            [355, 356.25], [358, 356.25],
            [358.7, 0], [359, 0], [360, 0], [361, 0],
            [362, 3.75], [363, 3.75], [364, 3.75], [365, 3.75], [366, 7.5],
            [-725.0, 356.25], [-722, 356.25], [-721, 0], [-719, 0.0],
            [-718, 3.75],
            [1234.56, 153.75], [-1234.56, 206.25]])
        lon_test_vals = lon_vals[:, 0]
        lon_expect_vals = lon_vals[:, 1]
        lon_coord_vals = self.cube.coord('longitude').points
        lon_expect_vals = [near_value(v, lon_coord_vals)
                           for v in lon_expect_vals]
        lon_nearest_inds = [
            iintrp.nearest_neighbour_indices(self.cube,
                                             [('longitude', point_val)])
            for point_val in lon_test_vals]
        lon_nearest_vals = [lon_coord_vals[i[1]] for i in lon_nearest_inds]
        self.assertArrayAlmostEqual(lon_nearest_vals, lon_expect_vals)


@tests.skip_data
class TestNearestNeighbour(tests.IrisTest, MixinNearestNeighbour):
    def setUp(self):
        self._common_setUp()


@tests.skip_data
class TestNearestNeighbour__Equivalent(tests.IrisTest, MixinNearestNeighbour):
    # Class that repeats the tests of "TestNearestNeighbour", to check that the
    # behaviour of the three 'nearest_neighbour' routines in
    # iris.analysis.interpolation can be completely replicated with alternative
    # (newer) functionality.

    def setUp(self):
        self.patch(
            'iris.analysis._interpolate_private.nearest_neighbour_indices',
            self._equivalent_nn_indices)
        self.patch(
            'iris.analysis._interpolate_private.nearest_neighbour_data_value',
            self._equivalent_nn_data_value)
        self.patch(
            'iris.analysis._interpolate_private.extract_nearest_neighbour',
            self._equivalent_extract_nn)
        self._common_setUp()

    @staticmethod
    def _equivalent_nn_indices(cube, sample_points,
                               require_single_point=False):
        indices = [slice(None) for _ in cube.shape]
        for coord_spec, point in sample_points:
            coord = cube.coord(coord_spec)
            dim, = cube.coord_dims(coord)  # expect only 1d --> single dim !
            dim_index = coord.nearest_neighbour_index(point)
            if require_single_point:
                # Mimic error behaviour of the original "data-value" function:
                # Any dim already addressed must get the same index.
                if indices[dim] != slice(None) and indices[dim] != dim_index:
                    raise ValueError('indices over-specified')
            indices[dim] = dim_index
        if require_single_point:
            # Mimic error behaviour of the original "data-value" function:
            # All dims must have an index.
            if any(index == slice(None) for index in indices):
                raise ValueError('result expected to be a single point')
        return tuple(indices)

    @staticmethod
    def _equivalent_extract_nn(cube, sample_points):
        indices = TestNearestNeighbour__Equivalent._equivalent_nn_indices(
            cube, sample_points)
        new_cube = cube[indices]
        return new_cube

    @staticmethod
    def _equivalent_nn_data_value(cube, sample_points):
        indices = TestNearestNeighbour__Equivalent._equivalent_nn_indices(
            # for this routine only, enable extra index checks.
            cube, sample_points, require_single_point=True)
        return cube.data[indices]


if __name__ == "__main__":
    tests.main()
