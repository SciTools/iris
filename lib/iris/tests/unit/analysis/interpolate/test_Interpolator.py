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
"""Unit tests for :class:`iris.analysis.interpolate.LinearInterpolator`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from nose.tools import assert_raises_regexp
import numpy as np

import iris
import iris.tests.stock as stock
from iris.analysis.new_linear import LinearInterpolator


class ThreeDimCube(tests.IrisTest):
    def setUp(self):
        cube = stock.simple_3d_w_multidim_coords()
        cube.add_aux_coord(iris.coords.DimCoord(range(2), 'height'), 0)
        cube.add_dim_coord(iris.coords.DimCoord(range(3), 'latitude'), 1)
        cube.add_dim_coord(iris.coords.DimCoord(range(4), 'longitude'), 2)
        self.cube = cube
        self.data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        
        cube.data = self.data


class Test_LinearInterpolator_1D(ThreeDimCube):
    def setUp(self):
        ThreeDimCube.setUp(self)
        self.interpolator = LinearInterpolator(self.cube, ['latitude'])

    def test_interpolate_data_single(self):
        assert np.all(self.interpolator.interpolate_data([1.5], self.data) ==
                      self.data[:, 1:3, :].mean(axis=1)[:, :, np.newaxis]), 'Wrong result'
    
    def test_interpolate_data_multiple(self):
        assert np.all(self.interpolator.interpolate_data([1, 2], self.data) ==
                      np.transpose(self.data[:, 1:3, :], [0, 2, 1])), 'Wrong result'
    
    def test_interpolate_data_linear_extrapolation(self):
        # Extrapolation (linear)
        expected = np.transpose(self.data[:, 0:1] - (self.data[:, 1:2] - self.data[:, 0:1]), [0, 2, 1])
        assert np.all(expected == self.interpolator.interpolate_data([-1], self.data)), 'Wrong result'
    
    def test_interpolate_data_nan_extrapolation(self):
        self.interpolator._interpolator.values = self.interpolator._interpolator.values.astype(np.float64)
        assert np.all(np.isnan(self.interpolator.interpolate_data([-1], self.data.astype(np.float64), extrapolation_mode='nan'))), 'Wrong result'
    
    def test_interpolate_data_nan_extrapolation_wrong_dtype(self):
        assert np.all(np.isnan(self.interpolator.interpolate_data([-1], self.data.astype(np.float64), extrapolation_mode='nan'))), 'Wrong result'
    
    def test_interpolate_data_error_on_extrapolation(self):
        with assert_raises_regexp(ValueError, 'One of the requested xi is out of bounds in dimension 0'):
            assert np.all(np.isnan(self.interpolator.interpolate_data([-1], self.data.astype(np.float64), extrapolation_mode='error'))), 'Wrong result'
    
    def test_bad_sample_points_array(self):
        with assert_raises_regexp(ValueError, 'The resulting numpy array has "object" as its type'):
            self.interpolator.interpolate_data([[1, 2], [1]], self.cube.data)
        
    def test_interpolate_data_dtype_casting(self):
        self.cube.data = self.cube.data.astype(int)
        interpolator = LinearInterpolator(self.cube, ['latitude'])
        self.assertEqual(interpolator.interpolate_data([0.125], self.data).dtype,
                         np.float32)

    def test_orthogonal_points(self):
        r = self.interpolator.orthogonal_points([['latitude', [1]]], self.cube.data) # TODO test it calls interpolate_data appropriately.
    
    def test_interpolate_data_with_data_dims(self):
        # Array spans 1 of the interpolation dimensions.
        assert np.all(np.transpose(self.data[0, 1].reshape(1, 1, -1), [0, 2, 1]) ==
                      self.interpolator.interpolate_data([1], self.data[0], data_dims=[1, 2]))
    
    def teet_inetrpolate_data_with_data_dims_not_interp(self):
        # No interpolation is taking place...
        self.assertArrayEqual(self.interpolator.interpolate_data([1], self.data[:, 0], data_dims=[0, 2]),
                              self.data[:, 0])


class SingleLengthDimension(ThreeDimCube):
    def setUp(self):
        ThreeDimCube.setUp(self)
        self.cube = self.cube[:, 0:1, 0]
        self.interpolator = LinearInterpolator(self.cube, ['latitude'])

    def test_interpolate_data_linear_extrapolation(self):
        # Linear extrapolation of a single valued element.
        assert np.all(self.interpolator.interpolate_data([1001], self.cube.data) ==
                      self.cube.data)
    
    def test_interpolate_data_nan_extrapolation(self):
        assert np.all(np.isnan(self.interpolator.interpolate_data([1001], self.cube.data,
                                                         extrapolation_mode='nan')))
    
    def test_interpolate_data_nan_extrapolation_not_needed(self):
        # No extrapolation for a single length dimension.
        assert np.all(self.interpolator.interpolate_data([0], self.cube.data,
                                                             extrapolation_mode='nan') ==
                      self.cube.data)

    def test_interpolator_overspecified(self):
        with self.assertRaisesRegexp(ValueError, 'Coordinates repeat a data dimension - the interpolation would be over-specified'):
            LinearInterpolator(self.cube, ['latitude', 'longitude'])


class Test_LinearInterpolator_monotonic(ThreeDimCube):
    def setUp(self):
        ThreeDimCube.setUp(self)
        self.cube = self.cube[:, ::-1]
        self.interpolator = LinearInterpolator(self.cube, ['latitude'])

    def test_interpolate_data(self):
        assert np.all(self.interpolator.interpolate_data([0], self.cube.data) ==
                      np.transpose(self.data[:, 0:1], [0, 2, 1])), 'Wrong result'


class Test_LinearInterpolator_circular(ThreeDimCube):
    def setUp(self):
        ThreeDimCube.setUp(self)
        self.cube.coord('longitude').points = np.linspace(0, 360, 4,
                                                          endpoint=False)
        self.cube.coord('longitude').circular = True
        self.cube.coord('longitude').units = 'degrees'
        self.interpolator = LinearInterpolator(self.cube, ['longitude'])

    def test_interpolate_data_fully_wrapped(self):
        expected = self.interpolator.interpolate_data([180, 270], self.cube.data, extrapolation_mode='nan')
        result = self.interpolator.interpolate_data([-180, -90], self.cube.data, extrapolation_mode='nan')
        self.assertArrayEqual(expected, result)
    
    def test_interpolate_data_partially_wrapped(self):
        expected = self.interpolator.interpolate_data([180, 90], self.cube.data, extrapolation_mode='nan')
        result = self.interpolator.interpolate_data([-180, 90], self.cube.data, extrapolation_mode='nan')
        self.assertArrayEqual(expected, result)

    def test_interpolate_data_fully_wrapped_twice(self):
        xs = np.linspace(-360, 360, 100)
        xs_not_wrapped = (xs + 360) % 360
        result = self.interpolator.interpolate_data(xs, self.cube.data)
        expected = self.interpolator.interpolate_data(xs_not_wrapped, self.cube.data)
        self.assertArrayEqual(expected, result)

# XXX Test Masked data...
        

class Test_LinearInterpolator_2D(ThreeDimCube):
    def setUp(self):
        ThreeDimCube.setUp(self)
        self.interpolator = LinearInterpolator(self.cube, ['latitude', 'longitude'])
    
    def test_interpolate_data(self):
        assert np.all(self.interpolator.interpolate_data([[1, 2], [2, 2]], self.data) ==
                      self.data[:, 1:3, 2:3].reshape(-1, 2)), 'Wrong result'

    def test_orthogonal_points(self):
        assert np.all(self.interpolator.orthogonal_points([['longitude', [1, 2]], ['latitude', [1, 2]]], self.cube.data) == 
                      self.interpolator.interpolate_data([[1, 1], [1, 2], [2, 1], [2, 2]],
                                                         self.cube.data).reshape(-1, 2, 2)), 'Wrong values'

    def test_interpolate_data_data_dims(self):
        assert np.all(self.interpolator.interpolate_data([0, 0], self.data[:, 0], data_dims=[0, 2]) ==
                      self.data[:, 0, 0].reshape(-1, 1))

    def test_interpolate_data_data_dims_multi_point(self):
        assert np.all(self.interpolator.interpolate_data([[0, 0], [1, 0], [2, 0]],
                                                         self.data[:, 0], data_dims=[0, 2]) ==
                      np.repeat(self.data[:, 0, 0].reshape(-1, 1), 3, axis=1))

    def test_orthogonal_points_data_dims(self):
        r = self.interpolator.orthogonal_points([['latitude', 0], ['longitude', 0]], self.data[:, 0], data_dims=[0, 2])
        assert np.all(r == np.array([[[0]], [[12]]]))

    def test_orthogonal_points_data_dims_multiple_points(self):
        r = self.interpolator.orthogonal_points([['latitude', [0, 1]], ['longitude', [0, 1, 1]]],
                                           self.data[:, 0],
                                           data_dims=[0, 2])
        
        expected = np.concatenate([self.data[:, 0:1, 0], self.data[:, 0:1, 0],
                                   self.data[:, 0:1, 1], self.data[:, 0:1, 1],
                                   self.data[:, 0:1, 1], self.data[:, 0:1, 1]],
                                  axis=1).reshape(2, 3, 2).transpose([0, 2, 1])
        assert np.all(r == expected)

    def test_orthogonal_points_data_dims_multiple_points_transposed(self):
        r = self.interpolator.orthogonal_points([['latitude', [0, 1, 1]], ['longitude', [0, 1]]],
                                           self.data[:, 0],
                                           data_dims=[0, 2])
        assert np.all(r == np.array([[[ 0,  1], [ 0,  1], [ 0,  1]],
                                     [[12, 13], [12, 13], [12, 13]]]))



class Test_LinearInterpolator_2D_non_contiguous(ThreeDimCube):
    def setUp(self):
        ThreeDimCube.setUp(self)
        self.interpolator = LinearInterpolator(self.cube, ['height', 'longitude'])

    def test_interpolate_data_multiple(self):
        assert np.all(self.interpolator.interpolate_data([[1, 1], [1, 2]], self.data) == 
                      self.data[1:2, :, 1:3]), 'Wrong result'
    
    def test_interpolate_data_single(self):
        assert np.all(self.interpolator.interpolate_data([[1, 1]], self.data) == 
                      self.interpolator.interpolate_data([1, 1], self.data))
    
    def test_interpolate_data_wrong_n_coordinates(self):
        with assert_raises_regexp(ValueError, 'coordinates are not appropriate for the interpolator'):
            self.interpolator.interpolate_data([1], self.data)


    def test_interpolate_data_wrong_shape_coordinates(self):
        with assert_raises_regexp(ValueError, ('There are 2 '
                                               r'dimension\(s\) and ideally the given coordinate array '
                                               r'should have a shape of \(npts\, 2\). Got an array of shape '
                                               r'\(2\, 4\)')):
            self.interpolator.interpolate_data([[1, 1, 2, 2], [1, 2, 1, 2]], self.cube.data)
    
    def test_intepolate_data_wrong_data_shape(self):
        with assert_raises_regexp(ValueError, 'data being interpolated is not consistent with the data passed through'):
            self.interpolator.interpolate_data([1], self.data[0])

    def test_interpolate_points_data_order(self):
        r = self.interpolator.orthogonal_points([['longitude', [0]], ['height', [0, 1]]], self.cube.data)
        r2 = self.interpolator.orthogonal_points([['height', [0, 1]], ['longitude', [0]]], self.cube.data)
        expected = self.cube.data[0:2, :, 0:1]
        assert np.all(r == expected), 'Wrong result'
        assert np.all(r == r2), 'Wrong result'
        
        inverse_interpolator = LinearInterpolator(self.cube, ['longitude', 'height'])
        r3 = inverse_interpolator.orthogonal_points([['longitude', [0]], ['height', [0, 1]]], self.cube.data)
        assert np.all(r == r3), 'Wrong result'
        r4 = inverse_interpolator.orthogonal_points([['height', [0, 1]], ['longitude', [0]]], self.cube.data)
        assert np.all(r == r4), 'Wrong result'

    def test_orthogonal_cube(self):
        result_cube = self.interpolator.orthogonal_cube([['height', np.int64([0, 1, 1])],
                                                         ['longitude', np.int32([0, 1])]])
        self.assertCMLApproxData(result_cube, ('experimental', 'analysis',
                                               'interpolate', 'LinearInterpolator', 'basic_orthogonal_cube.cml'))
        self.assertEqual(result_cube.coord('longitude').dtype, np.int32)
        self.assertEqual(result_cube.coord('height').dtype, np.int64)
    
    def test_orthogonal_cube_squash(self):
        result_cube = self.interpolator.orthogonal_cube([['height', np.int64(0)],
                                                         ['longitude', np.int32([0, 1])]])
        self.assertCMLApproxData(result_cube, ('experimental', 'analysis',
                                               'interpolate', 'LinearInterpolator', 'orthogonal_cube_1d_squashed.cml'))
        self.assertEqual(result_cube.coord('longitude').dtype, np.int32)
        self.assertEqual(result_cube.coord('height').dtype, np.int64)
        
        non_collapsed_cube = self.interpolator.orthogonal_cube([['height', np.int64(0)],
                                                                ['longitude', np.int32([0, 1])]], collapse_scalar=False)
        self.assertEqual(result_cube, non_collapsed_cube[0, ...])


if __name__ == "__main__":
    tests.main()
