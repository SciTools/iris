# (C) British Crown Copyright 2010 - 2014, Met Office
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
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import biggus
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d

import iris
import iris.coord_systems
import iris.cube
import iris.analysis.interpolate
import iris.tests.stock
from iris.analysis.interpolate import Linear1dExtrapolator
import iris.analysis.interpolate as iintrp


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
        r = Linear1dExtrapolator(interp1d(np.arange(3), a, axis=0))
        
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
        r = Linear1dExtrapolator(interp1d(np.arange(4), a, axis=1))
        
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
                                         np.array([[ -1.5,  -1. ,   1.5,   2. ,   4.5,   5. ],
                                                      [  2.5,   3. ,   5.5,   6. ,   8.5,   9. ],
                                                      [  6.5,   7. ,   9.5,  10. ,  12.5,  13. ]]))
        
        
    def test_simple_3d_axis1(self):
        a = np.arange(24.).reshape(3, 4, 2)
        r = Linear1dExtrapolator(interp1d(np.arange(4.), a, axis=1))
        
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
        r = Linear1dExtrapolator(interp1d(np.arange(2), a, axis=0))
        
        np.testing.assert_array_equal(r(0), np.array([ 2.,  4.,  8.]))
        np.testing.assert_array_equal(r(-1), np.array([ 4.,  3.,  5.]))
        np.testing.assert_array_equal(r(3), np.array([ -4.,   7.,  17.]))        
        np.testing.assert_array_equal(r(2.5), np.array([ -3. ,   6.5,  15.5]))
        
        np.testing.assert_array_equal(r(np.array([1.5, 2])), np.array([[ -1. ,   5.5,  12.5],
                                                                                [ -2. ,   6. ,  14. ]]))
    
        np.testing.assert_array_equal(r(np.array([-1.5, 3.5])), np.array([[  5. ,   2.5,   3.5],
                                                                                   [ -5. ,   7.5,  18.5]]))


class TestLinearLengthOneCoord(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.lat_lon_cube()
        self.cube.data = self.cube.data.astype(float)

    def test_single_point(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        r = iris.analysis.interpolate.linear(cube, [('longitude', [1.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_single_pt_0'))

        # Slice to form (1, 4) shaped cube.
        cube = self.cube[1:2, :]
        r = iris.analysis.interpolate.linear(cube, [('latitude', [1.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_single_pt_1'))

    def test_multiple_points(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        r = iris.analysis.interpolate.linear(cube, [('longitude',
                                                     [1., 2., 3., 4.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_0'))

        # Slice to form (1, 4) shaped cube.
        cube = self.cube[1:2, :]
        r = iris.analysis.interpolate.linear(cube, [('latitude',
                                                     [1., 2., 3., 4.])])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_1'))

    def test_single_point_to_scalar(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        r = iris.analysis.interpolate.linear(cube, [('longitude', 1.)])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_scalar_0'))

        # Slice to form (1, 4) shaped cube.
        cube = self.cube[1:2, :]
        r = iris.analysis.interpolate.linear(cube, [('latitude', 1.)])
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_scalar_1'))

    def test_extrapolation_mode_same_pt(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        src_points = cube.coord('longitude').points
        r = iris.analysis.interpolate.linear(cube, [('longitude', src_points)],
                                             extrapolation_mode='linear')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_same_pt'))
        r = iris.analysis.interpolate.linear(cube, [('longitude', src_points)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_same_pt'))
        r = iris.analysis.interpolate.linear(cube, [('longitude', src_points)],
                                             extrapolation_mode='error')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_same_pt'))

    def test_extrapolation_mode_multiple_same_pts(self):
        # Slice to form (3, 1) shaped cube.
        cube = self.cube[:, 2:3]
        src_points = cube.coord('longitude').points
        new_points = [src_points[0]] * 3
        r = iris.analysis.interpolate.linear(cube, [('longitude', new_points)],
                                             extrapolation_mode='linear')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_same'))
        r = iris.analysis.interpolate.linear(cube, [('longitude', new_points)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_same'))
        r = iris.analysis.interpolate.linear(cube, [('longitude', new_points)],
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
        r = iris.analysis.interpolate.linear(cube, [('longitude',
                                                     new_points_single)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_single_pt_nan'))
        r = iris.analysis.interpolate.linear(cube, [('longitude',
                                                     new_points_multiple)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_many_nan'))
        r = iris.analysis.interpolate.linear(cube, [('longitude',
                                                     new_points_scalar)],
                                             extrapolation_mode='nan')
        self.assertCMLApproxData(r, ('analysis', 'interpolation', 'linear',
                                     'single_pt_to_scalar_nan'))

        # 'error' mode
        with self.assertRaises(ValueError):
            r = iris.analysis.interpolate.linear(cube, [('longitude',
                                                         new_points_single)],
                                                 extrapolation_mode='error')
        with self.assertRaises(ValueError):
            r = iris.analysis.interpolate.linear(cube, [('longitude',
                                                         new_points_multiple)],
                                                 extrapolation_mode='error')
        with self.assertRaises(ValueError):
            r = iris.analysis.interpolate.linear(cube, [('longitude',
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
        r = iris.analysis.interpolate.linear(cube, [('dim1', [7, 3, 5])])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'dim_to_aux.cml'))

    def test_bad_sample_point_format(self):
        self.assertRaises(TypeError, iris.analysis.interpolate.linear, self.simple2d_cube, ('dim1', 4))
    
    def test_simple_single_point(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 4)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_single_point.cml'), checksum=False)
        np.testing.assert_array_equal(r.data, np.array([1.5, 2.5, 3.5], dtype=self.simple2d_cube.data.dtype))
        
    def test_monotonic_decreasing_coord(self):
        c = self.simple2d_cube[::-1]
        r = iris.analysis.interpolate.linear(c, [('dim1', 4)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_single_point.cml'), checksum=False)
        np.testing.assert_array_equal(r.data, np.array([1.5, 2.5, 3.5], dtype=self.simple2d_cube.data.dtype))
        
    def test_overspecified(self):
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, self.simple2d_cube[0, :], [('dim1', 4)])
        
    def test_bounded_coordinate(self):
        # The results should be exactly the same as for the
        # non-bounded case.
        cube = self.simple2d_cube
        cube.coord('dim1').guess_bounds()
        r = iris.analysis.interpolate.linear(cube, [('dim1', [4, 5])])
        np.testing.assert_array_equal(r.data, np.array([[ 1.5,  2.5,  3.5], [ 3. ,  4. ,  5. ]]))
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))

    def test_simple_multiple_point(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', [4, 5])])
        np.testing.assert_array_equal(r.data, np.array([[ 1.5,  2.5,  3.5], [ 3. ,  4. ,  5. ]]))
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))
        
        # Check that numpy arrays specifications work
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', np.array([4, 5]))])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))

    def test_circular_vs_non_circular_coord(self):
        cube = self.simple2d_cube_circular
        other = iris.coords.AuxCoord([10, 6, 7, 4], long_name='other')
        cube.add_aux_coord(other, 1)
        samples = [0, 60, 300]
        r = iris.analysis.interpolate.linear(cube, [('theta', samples)])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'circular_vs_non_circular.cml'))

    def test_simple_multiple_points_circular(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube_circular, [('theta', [0. , 60. , 120. , 180. ])])
        normalise_order(r)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points_circular.cml'))
        
        # check that the values returned by theta 0 & 360 are the same...
        r1 = iris.analysis.interpolate.linear(self.simple2d_cube_circular, [('theta', 360)])
        r2 = iris.analysis.interpolate.linear(self.simple2d_cube_circular, [('theta', 0)])
        np.testing.assert_array_almost_equal(r1.data, r2.data)
        
    def test_simple_multiple_coords(self):
        expected_result = np.array(2.5)
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 4), ('dim2', 3.5), ])
        np.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
        
        # Check that it doesn't matter if you do the interpolation in separate steps...
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim2', 3.5)])
        r = iris.analysis.interpolate.linear(r, [('dim1', 4)])
        np.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
        
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 4)])
        r = iris.analysis.interpolate.linear(r, [('dim2', 3.5)])
        np.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
    
    def test_coord_not_found(self):
        self.assertRaises(KeyError, iris.analysis.interpolate.linear, self.simple2d_cube, 
                          [('non_existant_coord', [3.5, 3.25])])
    
    def test_simple_coord_error_extrapolation(self):
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='error')

    def test_simple_coord_linear_extrapolation(self):
        r = iris.analysis.interpolate.linear( self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='linear')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_linear_extrapolation.cml'))
        
        np.testing.assert_array_equal(r.data, np.array([-1.,  2.,  5.,  8.], dtype=self.simple2d_cube.data.dtype))
        
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 1)])
        np.testing.assert_array_equal(r.data, np.array([-3., -2., -1.], dtype=self.simple2d_cube.data.dtype))
                
    def test_simple_coord_linear_extrapolation_multipoint1(self):
        r = iris.analysis.interpolate.linear( self.simple2d_cube, [('dim1', [-1, 1, 10, 11])], extrapolation_mode='linear')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_linear_extrapolation_multipoint1.cml'))
        
    def test_simple_coord_linear_extrapolation_multipoint2(self):
        r = iris.analysis.interpolate.linear( self.simple2d_cube, [('dim1', [1, 10])], extrapolation_mode='linear')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_linear_extrapolation_multipoint2.cml'))
        
    def test_simple_coord_nan_extrapolation(self):
        r = iris.analysis.interpolate.linear( self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='nan')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_nan_extrapolation.cml'))
    
    def test_multiple_coord_extrapolation(self):
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, self.simple2d_cube, [('dim2', 2.5), ('dim1', 12.5)], extrapolation_mode='error')    
        
    def test_multiple_coord_linear_extrapolation(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim2', 9), ('dim1', 1.5)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords_extrapolation.cml'))
        
    def test_lots_of_points(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', np.linspace(3, 9, 20))])
        # XXX Implement a test!?!
        
    def test_shared_axis(self):
        c = self.simple2d_cube_extended
        r = iris.analysis.interpolate.linear(c, [('dim2', [3.5, 3.25])])
        normalise_order(r)
        
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_shared_axis.cml'))
        
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, c, [('dim2', [3.5, 3.25]), ('shared_x_coord', [9, 7])])

    def test_points_datatype_casting(self):
        # this test tries to extract a float from an array of type integer. the result should be of type float.
        r = iris.analysis.interpolate.linear(self.simple2d_cube_extended, [('shared_x_coord', 7.5)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_casting_datatype.cml'))
        
    def test_mask(self):
        # Test np.append bug with masked arrays.
        # Based on the bug reported in https://github.com/SciTools/iris/issues/106
        cube = tests.stock.realistic_4d_w_missing_data()
        cube = cube[0, 2, 18::-1]
        cube.coord('grid_longitude').circular = True
        _ = iris.analysis.interpolate.linear(cube, [('grid_longitude',0), ('grid_latitude',0)])
        # Did np.append go wrong?
        self.assertArrayEqual(cube.data.data.shape, cube.data.mask.shape)
    
    def test_scalar_mask(self):
        # Testing the bug raised in https://github.com/SciTools/iris/pull/123#issuecomment-9309872
        # (the fix workaround for the np.append bug failed for scalar masks) 
        cube = tests.stock.realistic_4d_w_missing_data()
        cube.data = ma.arange(np.product(cube.shape), dtype=np.float32).reshape(cube.shape)
        cube.coord('grid_longitude').circular = True
        # There's no result to test, just make sure we don't cause an exception with the scalar mask.
        _ = iris.analysis.interpolate.linear(cube, [('grid_longitude',0), ('grid_latitude',0)])


@tests.skip_data
class TestNearestLinearInterpolRealData(tests.IrisTest):
    def setUp(self):
        file = tests.get_data_path(('PP', 'globClim1', 'theta.pp'))
        self.cube = iris.load_cube(file)

    def test_slice(self):
        r = iris.analysis.interpolate.linear(self.cube, [('latitude', 0)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'real_2dslice.cml'))
    
    def test_2slices(self):
        r = iris.analysis.interpolate.linear(self.cube, [('latitude', 0.0), ('longitude', 0.0)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'real_2slices.cml'))

    def test_circular(self):
        res = iris.analysis.interpolate.linear(self.cube,
                                               [('longitude', 359.8)])
        normalise_order(res)
        lon_coord = self.cube.coord('longitude').points
        expected = self.cube.data[..., 0] + \
            ((self.cube.data[..., -1] - self.cube.data[..., 0]) *
             (((360 - 359.8) - lon_coord[0]) /
              ((360 - lon_coord[-1]) - lon_coord[0])))
        self.assertArrayAllClose(res.data, expected, rtol=1.0e-6)

        # check that the values returned by lon 0 & 360 are the same...
        r1 = iris.analysis.interpolate.linear(self.cube, [('longitude', 360)])
        r2 = iris.analysis.interpolate.linear(self.cube, [('longitude', 0)])
        np.testing.assert_array_equal(r1.data, r2.data)

        self.assertCML(res, ('analysis', 'interpolation', 'linear',
                             'real_circular_2dslice.cml'), checksum=False)


@tests.skip_data
class TestNearestNeighbour(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.global_pp()
        points = np.arange(self.cube.coord('latitude').shape[0], dtype=np.float32)
        coord_to_add = iris.coords.DimCoord(points, long_name='i', units='meters')
        self.cube.add_aux_coord(coord_to_add, 0)

    def test_nearest_neighbour(self):
        point_spec = [('latitude', 40), ('longitude', 39)]
        
        indices = iris.analysis.interpolate.nearest_neighbour_indices(self.cube, point_spec)
        self.assertEqual(indices, (20, 10))
        
        b = iris.analysis.interpolate.extract_nearest_neighbour(self.cube, point_spec) 

        # Check that the data has not been loaded on either the original cube,
        # nor the interpolated one.
        self.assertTrue(b.has_lazy_data())
        self.assertTrue(self.cube.has_lazy_data())
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude_longitude.cml'))
        
        value = iris.analysis.interpolate.nearest_neighbour_data_value(self.cube, point_spec)
        self.assertEqual(value, np.array(285.98785, dtype=np.float32))

        # Check that the value back is that which was returned by the extract method
        self.assertEqual(value, b.data)
        
    def test_nearest_neighbour_slice(self):
        point_spec = [('latitude', 40)]
        indices = iris.analysis.interpolate.nearest_neighbour_indices(self.cube, point_spec)
        self.assertEqual(indices, (20, slice(None, None)))

        b = iris.analysis.interpolate.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude.cml'))
        
        # cannot get a specific point from these point specifications
        self.assertRaises(ValueError, iris.analysis.interpolate.nearest_neighbour_data_value, self.cube, point_spec)

    def test_nearest_neighbour_over_specification_which_is_consistent(self):
        # latitude 40 is the 20th point
        point_spec = [('latitude', 40), ('i', 20), ('longitude', 38)]
        
        indices = iris.analysis.interpolate.nearest_neighbour_indices(self.cube, point_spec)
        self.assertEqual(indices, (20, 10))
        
        b = iris.analysis.interpolate.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude_longitude.cml'))
        
        value = iris.analysis.interpolate.nearest_neighbour_data_value(self.cube, point_spec)
        # Check that the value back is that which was returned by the extract method
        self.assertEqual(value, b.data)

    def test_nearest_neighbour_over_specification_mis_aligned(self):
        # latitude 40 is the 20th point
        point_spec = [('latitude', 40), ('i', 10), ('longitude', 38)]
        
        # assert that we get a ValueError for over specifying our interpolation
        self.assertRaises(ValueError, iris.analysis.interpolate.nearest_neighbour_data_value, self.cube, point_spec)

    def test_nearest_neighbour_bounded_simple(self):
        point_spec = [('latitude', 37), ('longitude', 38)]
    
        coord = self.cube.coord('latitude')
        coord.guess_bounds(0.5)
    
        b = iris.analysis.interpolate.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_bounded.cml'))
        
    def test_nearest_neighbour_bounded_requested_midpoint(self):
        # This test checks the "point inside cell" logic
        point_spec = [('latitude', 38), ('longitude', 38)]
    
        coord = self.cube.coord('latitude')
        coord.guess_bounds(0.5)
        
        b = iris.analysis.interpolate.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_bounded_mid_point.cml'))
    
    def test_nearest_neighbour_locator_style_coord(self):
        point_spec = [('latitude', 39)]
        
        b = iris.analysis.interpolate.extract_nearest_neighbour(self.cube, point_spec) 
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


class TestNearestNeighbourAdditional(tests.IrisTest):
    """
    More detailed testing for coordinate nearest_neighbour function.

    Includes (especially) circular operation.

    """
    def _test_nn_breakpoints(self, points, breaks, expected,
                             bounds=None, guess_bounds=False, bounds_point=0.2,
                             circular=False,
                             test_min=-500.0, test_max=750.0):
        """
        Make a test coordinate + test the nearest-neighbour calculation

        Check that the result index changes at the specified points.
        Includes support for circular and bounded cases.

        Args:

            points : 1-d array-like
                Points of the test coordinate (see also Kwargs)
            breaks : list
                Input points at which we expect the output result to change
            expected : list
                Expected results (cell index values)
                Length must == len(breaks) + 1
                result == expected[i] between breaks[i] and breaks[i+1]

        Kwargs:

            bounds : 2d array-like
                use these bounds
            guess_bounds, bounds_point : bool, float
                use guessed bounds
            circular : bool
                make a circular coordinate (360 degrees)
            test_min, test_max : float
                outer extreme values (non-circular only)

        """
        points = np.array(points, np.float)
        if bounds:
            bounds = np.array(bounds, np.float)
        assert len(expected) == len(breaks) + 1
        if circular:
            breaks = np.array(breaks)
            breaks = np.hstack([i * 360.0 + breaks for i in range(-2, 3)])
            lower_lims = breaks[:-1]
            upper_lims = breaks[1:]
            expected = np.hstack([expected[1:] for i in range(-2, 3)])
        else:
            lower_lims = np.hstack(([test_min], breaks))
            upper_lims = np.hstack((breaks, [test_max]))

        # construct coord : AuxCoord, or DimCoord if it needs to be circular
        if circular:
            test_coord = iris.coords.DimCoord(points,
                                              bounds=bounds,
                                              long_name='x',
                                              units=iris.unit.Unit('degrees'),
                                              circular=True)
        else:
            test_coord = iris.coords.AuxCoord(points,
                                              bounds=bounds,
                                              long_name='x',
                                              units=iris.unit.Unit('degrees'))
        if guess_bounds:
            test_coord.guess_bounds(bounds_point)

        # test at a few 'random' points within each supposed result region
        test_fractions = np.array([0.01, 0.2, 0.45, 0.75, 0.99])
        for (lower, upper, expect) in zip(lower_lims, upper_lims, expected):
            test_pts = lower + test_fractions * (upper - lower)
            results = [test_coord.nearest_neighbour_index(x) for x in test_pts]
            self.assertTrue(np.all([r == expect for r in results]))

    def test_nearest_neighbour_circular(self):
        # First test (simplest): ascending-order, unbounded
        points = [0.0, 90.0, 180.0, 270.0]
        breaks = [45.0, 135.0, 225.0]
        results = [0, 1, 2, 3]
        self._test_nn_breakpoints(points, breaks, results)

        # same, but *CIRCULAR*
        breaks_circ = [-45.0] + breaks
        results_circ = [3] + results
        self._test_nn_breakpoints(points, breaks_circ, results_circ,
                                  circular=True)

        # repeat circular test with different coordinate offsets
        offset = 32.7
        points_offset = np.array(points) + offset
        breaks_offset = np.array(breaks_circ) + offset
        self._test_nn_breakpoints(points_offset, breaks_offset, results_circ,
                                  circular=True)

        offset = -106.3
        points_offset = np.array(points) + offset
        breaks_offset = np.array(breaks_circ) + offset
        self._test_nn_breakpoints(points_offset, breaks_offset, results_circ,
                                  circular=True)

        # ascending order, guess-bounded
        # N.B. effect of bounds_position = 2/3
        #   x_bounds = [[-60, 30], [30, 120], [120, 210], [210, 300]]
        points = [0.0, 90, 180, 270]
        breaks = [30.0, 120.0, 210.0]
        results = [0, 1, 2, 3]
        self._test_nn_breakpoints(points, breaks, results,
                                  guess_bounds=True, bounds_point=2.0 / 3)

        # same but circular...
        breaks_circ = [-60.0] + breaks
        results_circ = [3] + results
        self._test_nn_breakpoints(points, breaks_circ, results_circ,
                                  guess_bounds=True, bounds_point=2.0 / 3,
                                  circular=True)

    def test_nearest_neighbour_descending_circular(self):
        # descending order, unbounded
        points = [270.0, 180, 90, 0]
        breaks = [45.0, 135.0, 225.0]
        results = [3, 2, 1, 0]
        self._test_nn_breakpoints(points, breaks, results)

        # same but circular...
        breaks = [-45.0] + breaks
        results = [0] + results
        self._test_nn_breakpoints(points, breaks, results, circular=True)

        # repeat circular test with different coordinate offsets
        offset = 32.7
        points_offset = np.array(points) + offset
        breaks_offset = np.array(breaks) + offset
        self._test_nn_breakpoints(points_offset, breaks_offset, results,
                                  circular=True)

        offset = -106.3
        points_offset = np.array(points) + offset
        breaks_offset = np.array(breaks) + offset
        self._test_nn_breakpoints(points_offset, breaks_offset, results,
                                  circular=True)

        # descending order, guess-bounded
        points = [250.0, 150, 50, -50]
        # N.B. equivalent effect of bounds_position = 0.4
        # x_bounds = [[290, 190], [190, 90], [90, -10], [-10, -110]]
        breaks = [-10.0, 90.0, 190.0]
        results = [3, 2, 1, 0]
        self._test_nn_breakpoints(points, breaks, results,
                                  guess_bounds=True, bounds_point=0.4)
        # same but circular...
        breaks = [-110.0] + breaks
        results = [0] + results
        self._test_nn_breakpoints(points, breaks, results,
                                  guess_bounds=True, bounds_point=0.4,
                                  circular=True)

    def test_nearest_neighbour_odd_bounds(self):
        # additional: test with overlapping bounds
        points = [0.0, 90, 180, 270]
        bounds = [[-90.0, 90], [0, 180], [90, 270], [180, 360]]
        breaks = [45.0, 135.0, 225.0]
        results = [0, 1, 2, 3]
        self._test_nn_breakpoints(points, breaks, results, bounds=bounds)

        # additional: test with disjoint bounds
        points = [40.0, 90, 150, 270]
        bounds = [[0, 60], [70, 90], [140, 200], [210, 360]]
        breaks = [65.0, 115.0, 205.0]
        results = [0, 1, 2, 3]
        self._test_nn_breakpoints(points, breaks, results, bounds=bounds)

    def test_nearest_neighbour_scalar(self):
        points = [1.0]
        breaks = []
        results = [0]
        self._test_nn_breakpoints(points, breaks, results)

    def test_nearest_neighbour_nonmonotonic(self):
        # a bounded example
        points = [3.0,  4.0,  1.0,  7.0, 10.0]
        bounds = [[2.5,  3.5],
                  [3.5,  4.5],
                  [0.5,  1.5],
                  [6.5,  7.5],
                  [9.5, 10.5]]
        breaks = [2.0, 3.5, 5.5, 8.5]
        results = [2, 0, 1, 3, 4]
        self._test_nn_breakpoints(points, breaks, results, bounds=bounds)

        # a pointwise example
        points = [3.0,  3.5,  1.0,  8.0, 12.0]
        breaks = [2.0, 3.25, 5.75, 10.0]
        results = [2, 0, 1, 3, 4]
        self._test_nn_breakpoints(points, breaks, results)

        # NOTE: no circular cases, as AuxCoords _cannot_ be circular.


if __name__ == "__main__":
    tests.main()
