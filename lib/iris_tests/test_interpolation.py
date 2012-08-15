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
Test the interpolation of Iris cubes.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy

import iris
import iris.coord_systems
import iris.cube
import iris.analysis.interpolate
import iris.tests.stock


class TestLinear1dInterpolation(tests.IrisTest):
    def setUp(self):
        data = numpy.arange(12., dtype=numpy.float32).reshape((4, 3))
        c2 = iris.cube.Cube(data)
           
        c2.long_name = 'test 2d dimensional cube'
        c2.units = 'kelvin'
        
        pts = 3 + numpy.arange(4, dtype=numpy.float32) * 2
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

        pts = 0.1 + numpy.arange(5, dtype=numpy.float32) * 0.1
        f = iris.coords.DimCoord(pts, long_name='r', units=1)
        g = iris.coords.DimCoord([0.0, 90.0, 180.0, 270.0], long_name='theta', units='degrees', circular=True)
        data = numpy.arange(20., dtype=numpy.float32).reshape((5, 4))
        c4 = iris.cube.Cube(data)
        c4.add_dim_coord(f, 0)
        c4.add_dim_coord(g, 1)
        self.simple2d_cube_circular = c4

    def test_integer_interpol(self):
        c = self.simple2d_cube
        c.data = c.data.astype(numpy.int16)
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, c, [('dim1', 4)])
        
    def test_bad_sample_point_format(self):
        self.assertRaises(TypeError, iris.analysis.interpolate.linear, self.simple2d_cube, ('dim1', 4))
    
    def test_simple_single_point(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 4)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_single_point.cml'))
        numpy.testing.assert_array_equal(r.data, numpy.array([1.5, 2.5, 3.5], dtype=self.simple2d_cube.data.dtype))
        
    def test_monotonic_decreasing_coord(self):
        c = self.simple2d_cube[::-1]
        r = iris.analysis.interpolate.linear(c, [('dim1', 4)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_single_point.cml'))
        numpy.testing.assert_array_equal(r.data, numpy.array([1.5, 2.5, 3.5], dtype=self.simple2d_cube.data.dtype))
        
    def test_overspecified(self):
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, self.simple2d_cube[0, :], [('dim1', 4)])
        
    def test_bounded_coordinate(self):
        # The results should be exactly the same as for the
        # non-bounded case.
        cube = self.simple2d_cube
        cube.coord('dim1').guess_bounds()
        r = iris.analysis.interpolate.linear(cube, [('dim1', [4, 5])])
        numpy.testing.assert_array_equal(r.data, numpy.array([[ 1.5,  2.5,  3.5], [ 3. ,  4. ,  5. ]]))
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))

    def test_simple_multiple_point(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', [4, 5])])
        numpy.testing.assert_array_equal(r.data, numpy.array([[ 1.5,  2.5,  3.5], [ 3. ,  4. ,  5. ]]))
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))
        
        # Check that numpy arrays specifications work
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', numpy.array([4, 5]))])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points.cml'))

    def test_simple_multiple_points_circular(self):
        r = iris.analysis.interpolate.linear(self.simple2d_cube_circular, [('theta', [0. , 60. , 120. , 180. ])])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_points_circular.cml'))
        
        # check that the values returned by theta 0 & 360 are the same...
        r1 = iris.analysis.interpolate.linear(self.simple2d_cube_circular, [('theta', 360)])
        r2 = iris.analysis.interpolate.linear(self.simple2d_cube_circular, [('theta', 0)])
        numpy.testing.assert_array_almost_equal(r1.data, r2.data)
        
    def test_simple_multiple_coords(self):
        expected_result = numpy.array(2.5)
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 4), ('dim2', 3.5), ])
        numpy.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
        
        # Check that it doesn't matter if you do the interpolation in separate steps...
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim2', 3.5)])
        r = iris.analysis.interpolate.linear(r, [('dim1', 4)])
        r.data = r.data.astype(numpy.float64)
        numpy.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
        
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 4)])
        r = iris.analysis.interpolate.linear(r, [('dim2', 3.5)])
        numpy.testing.assert_array_equal(r.data, expected_result)
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_multiple_coords.cml'), checksum=False)
    
    def test_coord_not_found(self):
        self.assertRaises(KeyError, iris.analysis.interpolate.linear, self.simple2d_cube, 
                          [('non_existant_coord', [3.5, 3.25])])
    
    def test_simple_coord_error_extrapolation(self):
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='error')

    def test_simple_coord_linear_extrapolation(self):
        r = iris.analysis.interpolate.linear( self.simple2d_cube, [('dim2', 2.5)], extrapolation_mode='linear')
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_coord_linear_extrapolation.cml'))
        
        numpy.testing.assert_array_equal(r.data, numpy.array([-1.,  2.,  5.,  8.], dtype=self.simple2d_cube.data.dtype))
        
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', 1)])
        numpy.testing.assert_array_equal(r.data, numpy.array([-3., -2., -1.], dtype=self.simple2d_cube.data.dtype))
                
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
        r = iris.analysis.interpolate.linear(self.simple2d_cube, [('dim1', numpy.linspace(3, 9, 20))])
        
    def test_shared_axis(self):
        c = self.simple2d_cube_extended
        r = iris.analysis.interpolate.linear(c, [('dim2', [3.5, 3.25])])
        
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_shared_axis.cml'))
        
        self.assertRaises(ValueError, iris.analysis.interpolate.linear, c, [('dim2', [3.5, 3.25]), ('shared_x_coord', [9, 7])])

    def test_points_datatype_casting(self):
        # this test tries to extract a float from an array of type integer. the result should be of type float.
        r = iris.analysis.interpolate.linear(self.simple2d_cube_extended, [('shared_x_coord', 7.5)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'simple_casting_datatype.cml'))
    

@iris.tests.skip_data
class TestNearestLinearInterpolRealData(tests.IrisTest):
    def setUp(self):
        file = tests.get_data_path(('PP', 'globClim1', 'theta.pp'))
        self.cube = iris.load_strict(file)

    def test_slice(self):
        r = iris.analysis.interpolate.linear(self.cube, [('latitude', 0)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'real_2dslice.cml'))
    
    def test_2slices(self):
        r = iris.analysis.interpolate.linear(self.cube, [('latitude', 0.0), ('longitude', 0.0)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'real_2slices.cml'))
    
    def test_circular(self):
        r = iris.analysis.interpolate.linear(self.cube, [('longitude', 359.8)])
        self.assertCML(r, ('analysis', 'interpolation', 'linear', 'real_circular_2dslice.cml'))
        
        # check that the values returned by lon 0 & 360 are the same...
        r1 = iris.analysis.interpolate.linear(self.cube, [('longitude', 360)])
        r2 = iris.analysis.interpolate.linear(self.cube, [('longitude', 0)])
        numpy.testing.assert_array_equal(r1.data, r2.data)


@iris.tests.skip_data    
class TestNearestNeighbour(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.global_pp()
        points = numpy.arange(self.cube.coord('latitude').shape[0], dtype=numpy.float32)
        coord_to_add = iris.coords.DimCoord(points, long_name='i', units='meters')
        self.cube.add_aux_coord(coord_to_add, 0)

    def test_nearest_neighbour(self):
        point_spec = [('latitude', 40), ('longitude', 39)]
        
        indices = iris.analysis.interpolate.nearest_neighbour_indices(self.cube, point_spec)
        self.assertEqual(indices, (20, 10))
        
        b = iris.analysis.interpolate.extract_nearest_neighbour(self.cube, point_spec) 
        self.assertCML(b, ('analysis', 'interpolation', 'nearest_neighbour_extract_latitude_longitude.cml'))
        
        value = iris.analysis.interpolate.nearest_neighbour_data_value(self.cube, point_spec)
        self.assertEqual(value, numpy.array(285.98785, dtype=numpy.float32))

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
    
    
if __name__ == "__main__":
    tests.main()
