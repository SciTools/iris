# (C) British Crown Copyright 2010 - 2012, Met Office
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
Test iris.util

"""
import numpy
import unittest
from scipy.interpolate import interp1d
import inspect

import iris.util


class TestMonotonic(unittest.TestCase):
    def assertMonotonic(self, array, direction=None, **kwargs):
        if direction is not None:
            mono, dir = iris.util.monotonic(array, return_direction=True, **kwargs)
            if not mono:
                self.fail('Array was not monotonic:/n %r' % array)
            if dir != numpy.sign(direction):
                self.fail('Array was monotonic but not in the direction expected:'
                          '/n  + requested direction: %s/n  + resultant direction: %s' % (direction, dir)) 
        else:
            mono = iris.util.monotonic(array, **kwargs)
            if not mono:
                self.fail('Array was not monotonic:/n %r' % array)
                
    def assertNotMonotonic(self, array, **kwargs):
        mono = iris.util.monotonic(array, **kwargs)
        if mono:
            self.fail("Array was monotonic when it shouldn't be:/n %r" % array)
        
    def test_monotonic_pve(self):
        a = numpy.array([3, 4, 5.3])
        self.assertMonotonic(a)
        self.assertMonotonic(a, direction=1)
        
        # test the reverse for negative monotonic.
        a = a[::-1]
        self.assertMonotonic(a)
        self.assertMonotonic(a, direction=-1)
        
    def test_not_monotonic(self):
        b = numpy.array([3, 5.3, 4])
        self.assertNotMonotonic(b)
        
    def test_monotonic_strict(self):
        b = numpy.array([3, 5.3, 4])
        self.assertNotMonotonic(b, strict=True)
        
        b = numpy.array([3, 5.3, 5.3])
        self.assertNotMonotonic(b, strict=True)
        self.assertMonotonic(b)
        
        b = numpy.array([0.0])
        self.assertRaises(ValueError, iris.util.monotonic, b)
        self.assertRaises(ValueError, iris.util.monotonic, b, strict=True)
        
        b = numpy.array([0.0, 0.0])
        self.assertNotMonotonic(b, strict=True)
        self.assertMonotonic(b)
         
class TestReverse(unittest.TestCase):
    def test_simple(self):
        a = numpy.arange(12).reshape(3, 4)
        numpy.testing.assert_array_equal(a[::-1], iris.util.reverse(a, 0))
        numpy.testing.assert_array_equal(a[::-1, ::-1], iris.util.reverse(a, [0, 1]))
        numpy.testing.assert_array_equal(a[:, ::-1], iris.util.reverse(a, 1))
        numpy.testing.assert_array_equal(a[:, ::-1], iris.util.reverse(a, [1]))
        self.assertRaises(ValueError, iris.util.reverse, a, [])
        self.assertRaises(ValueError, iris.util.reverse, a, -1)
        self.assertRaises(ValueError, iris.util.reverse, a, 10)
        self.assertRaises(ValueError, iris.util.reverse, a, [-1])
        self.assertRaises(ValueError, iris.util.reverse, a, [0, -1])
        
    def test_single(self):
        a = numpy.arange(36).reshape(3, 4, 3)
        numpy.testing.assert_array_equal(a[::-1], iris.util.reverse(a, 0))
        numpy.testing.assert_array_equal(a[::-1, ::-1], iris.util.reverse(a, [0, 1]))
        numpy.testing.assert_array_equal(a[:, ::-1, ::-1], iris.util.reverse(a, [1, 2]))
        numpy.testing.assert_array_equal(a[..., ::-1], iris.util.reverse(a, 2))
        self.assertRaises(ValueError, iris.util.reverse, a, -1)
        self.assertRaises(ValueError, iris.util.reverse, a, 10)
        self.assertRaises(ValueError, iris.util.reverse, a, [-1])
        self.assertRaises(ValueError, iris.util.reverse, a, [0, -1])
        
        
class TestLinearExtrapolator(unittest.TestCase):
    def test_simple_axis0(self):
        a = numpy.arange(12.).reshape(3, 4)
        r = iris.util.Linear1dExtrapolator(interp1d(numpy.arange(3), a, axis=0))
        
        numpy.testing.assert_array_equal(r(0), numpy.array([ 0.,  1.,  2.,  3.]))
        numpy.testing.assert_array_equal(r(-1), numpy.array([-4.,  -3.,  -2., -1.]))
        numpy.testing.assert_array_equal(r(3), numpy.array([ 12.,  13.,  14.,  15.]))        
        numpy.testing.assert_array_equal(r(2.5), numpy.array([ 10.,  11.,  12.,  13.]))
        
        # 2 Non-extrapolation point
        numpy.testing.assert_array_equal(r(numpy.array([1.5, 2])), numpy.array([[  6.,   7.,   8.,   9.],
                                                                                [  8.,   9.,  10.,  11.]]))
        
        # 1 Non-extrapolation point & 1 upper value extrapolation
        numpy.testing.assert_array_equal(r(numpy.array([1.5, 3])), numpy.array([[  6.,   7.,   8.,   9.],
                                                                                [ 12.,  13.,  14.,  15.]]))
        
        # 2 upper value extrapolation
        numpy.testing.assert_array_equal(r(numpy.array([2.5, 3])), numpy.array([[ 10.,  11.,  12.,  13.],
                                                                                [ 12.,  13.,  14.,  15.]]))

        # 1 lower value extrapolation & 1 Non-extrapolation point
        numpy.testing.assert_array_equal(r(numpy.array([-1, 1.5])), numpy.array([[-4., -3., -2., -1.],
                                                                                 [ 6.,  7.,  8.,  9.]]))

        # 2 lower value extrapolation
        numpy.testing.assert_array_equal(r(numpy.array([-1.5, -1])), numpy.array([[-6., -5., -4., -3.],
                                                                                  [-4., -3., -2., -1.]]))
        
        # 2 lower value extrapolation, 2 Non-extrapolation point & 2 upper value extrapolation
        numpy.testing.assert_array_equal(r(numpy.array([-1.5, -1, 1, 1.5, 2.5, 3])),
                                         numpy.array([[ -6.,  -5.,  -4.,  -3.],
                                                      [ -4.,  -3.,  -2.,  -1.],
                                                      [  4.,   5.,   6.,   7.],
                                                      [  6.,   7.,   8.,   9.],
                                                      [ 10.,  11.,  12.,  13.],
                                                      [ 12.,  13.,  14.,  15.]]))
   
    def test_simple_axis1(self):
        a = numpy.arange(12).reshape(3, 4)
        r = iris.util.Linear1dExtrapolator(interp1d(numpy.arange(4), a, axis=1))
        
        # check non-extrapolation given the Extrapolator object 
        numpy.testing.assert_array_equal(r(0), numpy.array([ 0.,  4.,  8.]))
        
        # check the result's shape in a 1d array (of len 0 & 1)
        numpy.testing.assert_array_equal(r(numpy.array(0)), numpy.array([ 0.,  4.,  8.]))        
        numpy.testing.assert_array_equal(r(numpy.array([0])), numpy.array([ [0.],  [4.],  [8.]]))
        
        # check extrapolation below the minimum value (and check the equivalent 0d & 1d arrays)
        numpy.testing.assert_array_equal(r(-1), numpy.array([-1., 3., 7.]))
        numpy.testing.assert_array_equal(r(numpy.array(-1)), numpy.array([-1., 3., 7.]))
        numpy.testing.assert_array_equal(r(numpy.array([-1])), numpy.array([[-1.], [ 3.], [ 7.]]))
        
        # check extrapolation above the maximum value
        numpy.testing.assert_array_equal(r(3), numpy.array([  3.,   7.,  11.]))
        numpy.testing.assert_array_equal(r(2.5), numpy.array([  2.5,   6.5,  10.5]))
        
        # 2 Non-extrapolation point 
        numpy.testing.assert_array_equal(r(numpy.array([1.5, 2])), numpy.array([[  1.5,   2. ],
                                                                                [  5.5,   6. ],
                                                                                [  9.5,  10. ]]))
        
        # 1 Non-extrapolation point & 1 upper value extrapolation
        numpy.testing.assert_array_equal(r(numpy.array([1.5, 5])), numpy.array([[  1.5,   5. ],
                                                                                [  5.5,   9. ],
                                                                                [  9.5,  13. ]]))
        
        # 2 upper value extrapolation        
        numpy.testing.assert_array_equal(r(numpy.array([4.5, 5])), numpy.array([[  4.5,   5. ],
                                                                                [  8.5,   9. ],
                                                                                [ 12.5,  13. ]]))

        # 1 lower value extrapolation & 1 Non-extrapolation point
        numpy.testing.assert_array_equal(r(numpy.array([-0.5, 1.5])), numpy.array([[-0.5,  1.5],
                                                                                   [ 3.5,  5.5],
                                                                                   [ 7.5,  9.5]]))        
        
        # 2 lower value extrapolation
        numpy.testing.assert_array_equal(r(numpy.array([-1.5, -1])), numpy.array([[-1.5, -1. ],
                                                                                  [ 2.5,  3. ],
                                                                                  [ 6.5,  7. ]]))
        
        # 2 lower value extrapolation, 2 Non-extrapolation point & 2 upper value extrapolation
        numpy.testing.assert_array_equal(r(numpy.array([-1.5, -1, 1.5, 2, 4.5, 5])), 
                                         numpy.array([[ -1.5,  -1. ,   1.5,   2. ,   4.5,   5. ],
                                                      [  2.5,   3. ,   5.5,   6. ,   8.5,   9. ],
                                                      [  6.5,   7. ,   9.5,  10. ,  12.5,  13. ]]))
        
        
    def test_simple_3d_axis1(self):
        a = numpy.arange(24.).reshape(3, 4, 2)
        r = iris.util.Linear1dExtrapolator(interp1d(numpy.arange(4.), a, axis=1))
        
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

        numpy.testing.assert_array_equal(r(0), numpy.array([[  0.,   1.],
                                                            [  8.,   9.],
                                                            [ 16.,  17.]]))
        
        numpy.testing.assert_array_equal(r(1), numpy.array([[  2.,   3.],
                                                            [ 10.,  11.],
                                                            [ 18.,  19.]]))
        
        numpy.testing.assert_array_equal(r(-1), numpy.array([[ -2.,  -1.],
                                                             [  6.,   7.],
                                                             [ 14.,  15.]]))
        
        numpy.testing.assert_array_equal(r(4), numpy.array([[  8.,   9.],
                                                            [ 16.,  17.],
                                                            [ 24.,  25.]]))

        numpy.testing.assert_array_equal(r(0.25), numpy.array([[  0.5,   1.5],
                                                               [  8.5,   9.5],
                                                               [ 16.5,  17.5]]))
        
        numpy.testing.assert_array_equal(r(-0.25), numpy.array([[ -0.5,   0.5],
                                                                [  7.5,   8.5],
                                                                [ 15.5,  16.5]]))
        
        numpy.testing.assert_array_equal(r(4.25), numpy.array([[  8.5,   9.5],
                                                               [ 16.5,  17.5],
                                                               [ 24.5,  25.5]]))
        
        numpy.testing.assert_array_equal(r(numpy.array([0.5, 1])), numpy.array([[[  1.,   2.], [  2.,   3.]],
                                                                                [[  9.,  10.], [ 10.,  11.]],
                                                                                [[ 17.,  18.], [ 18.,  19.]]]))
        
        numpy.testing.assert_array_equal(r(numpy.array([0.5, 4])), numpy.array([[[  1.,   2.], [  8.,   9.]],
                                                                                [[  9.,  10.], [ 16.,  17.]],
                                                                                [[ 17.,  18.], [ 24.,  25.]]]))

        numpy.testing.assert_array_equal(r(numpy.array([-0.5, 0.5])), numpy.array([[[ -1.,   0.], [  1.,   2.]],
                                                                                   [[  7.,   8.], [  9.,  10.]],
                                                                                   [[ 15.,  16.], [ 17.,  18.]]]))        

        numpy.testing.assert_array_equal(r(numpy.array([-1.5, -1, 0.5, 1, 4.5, 5])), 
                                         numpy.array([[[ -3.,  -2.], [ -2.,  -1.], [  1.,   2.], [  2.,   3.], [  9.,  10.], [ 10.,  11.]],
                                                      [[  5.,   6.], [  6.,   7.], [  9.,  10.], [ 10.,  11.], [ 17.,  18.], [ 18.,  19.]],
                                                      [[ 13.,  14.], [ 14.,  15.], [ 17.,  18.], [ 18.,  19.], [ 25.,  26.], [ 26.,  27.]]]))
        
    def test_variable_gradient(self):
        a = numpy.array([[2, 4, 8], [0, 5, 11]])
        r = iris.util.Linear1dExtrapolator(interp1d(numpy.arange(2), a, axis=0))
        
        numpy.testing.assert_array_equal(r(0), numpy.array([ 2.,  4.,  8.]))
        numpy.testing.assert_array_equal(r(-1), numpy.array([ 4.,  3.,  5.]))
        numpy.testing.assert_array_equal(r(3), numpy.array([ -4.,   7.,  17.]))        
        numpy.testing.assert_array_equal(r(2.5), numpy.array([ -3. ,   6.5,  15.5]))
        
        numpy.testing.assert_array_equal(r(numpy.array([1.5, 2])), numpy.array([[ -1. ,   5.5,  12.5],
                                                                                [ -2. ,   6. ,  14. ]]))
    
        numpy.testing.assert_array_equal(r(numpy.array([-1.5, 3.5])), numpy.array([[  5. ,   2.5,   3.5],
                                                                                   [ -5. ,   7.5,  18.5]]))

class TestClipString(unittest.TestCase):

    def setUp(self):
        self.test_string = "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        self.rider = "**^^**$$..--__" # A good chance at being unique and not in the string to be tested!

    def test_oversize_string(self):
        # Test with a clip length that means the string will be clipped
        
        clip_length = 109
        result = iris.util.clip_string(self.test_string, clip_length, self.rider)
        
        # Check the length is between what we requested ( + rider length) and the length of the original string
        self.assertTrue(clip_length + len(self.rider) <= len(result) < len(self.test_string), "String was not clipped.")
        
        # Also test the rider was added
        self.assertTrue(self.rider in result, "Rider was not added to the string when it should have been.")
        
    def test_undersize_string(self):
        # Test with a clip length that is longer than the string
        
        clip_length = 10999
        result = iris.util.clip_string(self.test_string, clip_length, self.rider)
        self.assertEqual(len(result), len(self.test_string), "String was clipped when it should not have been.")
        
        # Also test that no rider was added on the end if the string was not clipped
        self.assertFalse(self.rider in result, "Rider was adding to the string when it should not have been.")
        
    def test_invalid_clip_lengths(self):
        # Clip values less than or equal to zero are not valid
        for clip_length in [0, -100]:    
            result = iris.util.clip_string(self.test_string, clip_length, self.rider)
            self.assertEqual(len(result), len(self.test_string), "String was clipped when it should not have been.")
        
    def test_default_values(self):
        # Get the default values specified in the function
        argspec = inspect.getargspec(iris.util.clip_string)
        arg_dict = dict(zip(argspec.args[-2:], argspec.defaults))

        result = iris.util.clip_string(self.test_string, arg_dict["clip_length"], arg_dict["rider"])
        
        self.assertLess(len(result), len(self.test_string), "String was not clipped.")
        
        rider_returned = result[-len(arg_dict["rider"]):]
        self.assertEquals(rider_returned, arg_dict["rider"], "Default rider was not applied.")
        
    def test_trim_string_with_no_spaces(self):

        clip_length = 200
        no_space_string = "a" * 500
        
        # Since this string has no spaces, clip_string will not be able to gracefully clip it
        # but will instead clip it exactly where the user specified
        result = iris.util.clip_string(no_space_string, clip_length, self.rider)
        
        expected_length = clip_length + len(self.rider)
        
        # Check the length of the returned string is equal to clip length + length of rider
        self.assertEquals(len(result), expected_length, "Mismatch in expected length of clipped string. Length was %s, expected value is %s" % (len(result), expected_length))
        
if __name__ == '__main__':
    unittest.main()
