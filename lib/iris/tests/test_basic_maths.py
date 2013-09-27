# (C) British Crown Copyright 2010 - 2013, Met Office
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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import operator

import numpy as np
import numpy.ma as ma

import iris
import iris.analysis.maths
import iris.coords
import iris.exceptions
import iris.tests.stock


@iris.tests.skip_data
class TestBasicMaths(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.global_pp()
        self.cube.data = self.cube.data - 260
    
    def test_abs(self):  
        a = self.cube                      
        
        b = iris.analysis.maths.abs(a, in_place=False)
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        self.assertCML(b, ('analysis', 'abs.cml'))
        
        iris.analysis.maths.abs(a, in_place=True)
        self.assertCML(b, ('analysis', 'abs.cml'))
        self.assertCML(a, ('analysis', 'abs.cml'))

    def test_minus(self):
        a = self.cube
        e = self.cube.copy()

        # Check that the subtraction has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))

        d = a - a
        self.assertCML(d, ('analysis', 'subtract.cml'))

        # Check that the subtraction has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        c = iris.analysis.maths.subtract(e, e)
        self.assertCML(c, ('analysis', 'subtract.cml'))
                        
        # Check that the subtraction has had no effect on the original
        self.assertCML(e, ('analysis', 'maths_original.cml'))
        
    def test_minus_in_need_of_transpose(self):
        a = self.cube
        e = self.cube.copy()
        e.transpose([1, 0])
        self.assertRaises(ValueError, iris.analysis.maths.subtract, a, e)
        
        
    def test_minus_with_data_describing_coordinate(self):
        a = self.cube
        e = self.cube.copy()
        lat = e.coord('latitude')
        lat.points = lat.points+100

        # Cannot ignore a axis describing coordinate
        self.assertRaises(ValueError, iris.analysis.maths.subtract, a, e)

    def test_minus_scalar(self):
        a = self.cube
        
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        b = a - 200
        self.assertCML(b, ('analysis', 'subtract_scalar.cml'))
        # Check that the subtraction has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
    def test_minus_array(self):
        a = self.cube
        data_array = self.cube.copy().data
        
        # check that the file has not changed (avoids false positives by failing early)
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        # subtract an array of exactly the same shape as the original
        b = a - data_array        
        self.assertArrayEqual(b.data, np.array(0, dtype=np.float32))
        self.assertCML(b, ('analysis', 'subtract_array.cml'), checksum=False)

        # subtract an array of the same number of dimensions, but with one of the dimensions having len 1
        b = a - data_array[:, 0:1]
        self.assertArrayEqual(b.data[:, 0:1], np.array(0, dtype=np.float32))
        self.assertArrayEqual(b.data[:, 1:2], b.data[:, 1:2])

        # subtract an array of 1 dimension fewer than the cube
        b = a - data_array[0, :]
        self.assertArrayEqual(b.data[0, :], np.array(0, dtype=np.float32))
        self.assertArrayEqual(b.data[:, 1:2], b.data[:, 1:2])
        
        # subtract an array of 1 dimension more than the cube
        d_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)
        self.assertRaises(ValueError, iris.analysis.maths.subtract, a, d_array)
                        
        # Check that the subtraction has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
    def test_minus_coord(self):
        a = self.cube

        xdim = a.ndim-1 
        ydim = a.ndim-2
        c_x = iris.coords.DimCoord(points=range(a.shape[xdim]), long_name='x_coord', units=self.cube.units) 
        c_y = iris.coords.AuxCoord(points=range(a.shape[ydim]), long_name='y_coord', units=self.cube.units) 
        
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        b = iris.analysis.maths.subtract(a, c_x, dim=1)
        self.assertCML(b, ('analysis', 'subtract_coord_x.cml'))
        # Check that the subtraction has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        b = iris.analysis.maths.subtract(a, c_y, dim=0)
        self.assertCML(b, ('analysis', 'subtract_coord_y.cml'))
        # Check that the subtraction has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
    def test_addition_scalar(self):
        a = self.cube
        
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        b = a + 200
        self.assertCML(b, ('analysis', 'addition_scalar.cml'))
        # Check that the addition has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
    def test_addition_coord(self):
        a = self.cube

        xdim = a.ndim-1 
        ydim = a.ndim-2
        c_x = iris.coords.DimCoord(points=range(a.shape[xdim]), long_name='x_coord', units=self.cube.units) 
        c_y = iris.coords.AuxCoord(points=range(a.shape[ydim]), long_name='y_coord', units=self.cube.units) 
        
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        b = iris.analysis.maths.add(a, c_x, dim=1)
        self.assertCML(b, ('analysis', 'addition_coord_x.cml'))
        # Check that the addition has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
        b = iris.analysis.maths.add(a, c_y, dim=0)
        self.assertCML(b, ('analysis', 'addition_coord_y.cml'))
        # Check that the addition has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
    
    def test_addition(self):
        a = self.cube

        c = a + a
        self.assertCML(c, ('analysis', 'addition.cml'))
        # Check that the addition has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
    def test_addition_different_standard_name(self):
        a = self.cube.copy()
        b = self.cube.copy()
        b.rename('my cube data')
        c = a + b        
        self.assertCML(c, ('analysis', 'addition_different_std_name.cml'), checksum=False)
        
    def test_addition_fail(self):
        a = self.cube
        
        xdim = a.ndim-1 
        ydim = a.ndim-2 
        c_axis_length_fail = iris.coords.DimCoord(points=range(a.shape[ydim]), long_name='x_coord', units=self.cube.units) 
        c_unit_fail = iris.coords.AuxCoord(points=range(a.shape[xdim]), long_name='x_coord', units='volts') 

        self.assertRaises(ValueError, iris.analysis.maths.add, a, c_axis_length_fail)
        self.assertRaises(iris.exceptions.NotYetImplementedError, iris.analysis.maths.add, a, c_unit_fail)

    def test_addition_in_place(self):
        a = self.cube

        b = iris.analysis.maths.add(a, self.cube, in_place=True)
        self.assertTrue(b is a)
        self.assertCML(a, ('analysis', 'addition_in_place.cml'))

    def test_addition_in_place_coord(self):
        a = self.cube

        # scalar is promoted to a coordinate internally
        b = iris.analysis.maths.add(a, 1000, in_place=True)
        self.assertTrue(b is a)
        self.assertCML(a, ('analysis', 'addition_in_place_coord.cml'))
        
    def test_addition_different_attributes(self):
        a = self.cube.copy()
        b = self.cube.copy()
        b.attributes['my attribute'] = 'foobar'
        c = a + b        
        self.assertIsNone(c.standard_name)
        self.assertEqual(c.attributes, {})


@iris.tests.skip_data
class TestDivideAndMultiply(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.global_pp()
        self.cube.data = self.cube.data - 260

    def test_divide(self):
        a = self.cube

        c = a / a
        
        np.testing.assert_array_almost_equal(a.data / a.data, c.data)
        self.assertCML(c, ('analysis', 'division.cml'), checksum=False)

        # Check that the division has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
    def test_divide_by_scalar(self):
        a = self.cube

        c = a / 10
        
        np.testing.assert_array_almost_equal(a.data / 10, c.data)
        self.assertCML(c, ('analysis', 'division_scalar.cml'), checksum=False)

        # Check that the division has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))

    def test_divide_by_coordinate(self):
        a = self.cube
                
        c = a / a.coord('latitude')
        self.assertCML(c, ('analysis', 'division_by_latitude.cml'))
        
        # Check that the division has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))
        
    def test_divide_by_array(self):
        a = self.cube
        data_array = self.cube.copy().data
        
        # test division by exactly the same shape data
        c = a / data_array 
        self.assertArrayEqual(c.data, np.array(1, dtype=np.float32))
        self.assertCML(c, ('analysis', 'division_by_array.cml'), checksum=False)
        
        # test division by array of fewer dimensions
        c = a / data_array[0, :] 
        self.assertArrayEqual(c.data[0, :], np.array(1, dtype=np.float32))
        
        # test division by array of more dimensions
        d_array = data_array.reshape(-1, data_array.shape[1], 1, 1)
        self.assertRaises(ValueError, iris.analysis.maths.divide, c, d_array) 
                
        # Check that the division has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))    
        
    def test_divide_by_coordinate_dim2(self):
        a = self.cube

        # Prevent divide-by-zero warning
        a.coord('longitude').points = a.coord('longitude').points + 0.5 

        c = a / a.coord('longitude')
        self.assertCML(c, ('analysis', 'division_by_longitude.cml'))

        # Reset to allow comparison with original
        a.coord('longitude').points = a.coord('longitude').points - 0.5 

        # Check that the division has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))

    def test_divide_by_singluar_coordinate(self):
        a = self.cube
        
        coord = iris.coords.DimCoord(points=2, long_name='foo', units='1')
        c = iris.analysis.maths.divide(a, coord)
        self.assertCML(c, ('analysis', 'division_by_singular_coord.cml'))
        
        # Check that the division is equivalent to dividing the whole of the data by 2 
        self.assertArrayEqual(c.data, a.data/2.)
        
    def test_divide_by_different_len_coord(self):
        a = self.cube
        
        coord = iris.coords.DimCoord(points=np.arange(10) * 2 + 5, standard_name='longitude', units='degrees') 
        
        self.assertRaises(ValueError, iris.analysis.maths.divide, a, coord)

    def test_divide_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.divide(a, 5, in_place=True)
        self.assertIs(a, b)

    def test_divide_not_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.divide(a, 5, in_place=False)
        self.assertIsNot(a, b)
        
    def test_multiply(self):
        a = self.cube
        
        c = a * a
        self.assertCML(c, ('analysis', 'multiply.cml'))
        
        # Check that the multiplication has had no effect on the original
        self.assertCML(a, ('analysis', 'maths_original.cml'))

    def test_multiplication_different_standard_name(self):
        a = self.cube.copy()
        b = self.cube.copy()
        b.rename('my cube data')
        c = a * b        
        self.assertCML(c, ('analysis', 'multiply_different_std_name.cml'), checksum=False)
        
    def test_multiplication_different_attributes(self):
        a = self.cube.copy()
        b = self.cube.copy()
        b.attributes['my attribute'] = 'foobar'
        c = a * b        
        self.assertIsNone(c.standard_name)
        self.assertEqual(c.attributes, {})

    def test_multiplication_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.multiply(a, 5, in_place=True)
        self.assertIs(a, b)

    def test_multiplication_not_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.multiply(a, 5, in_place=False)
        self.assertIsNot(a, b)


@iris.tests.skip_data
class TestExponentiate(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.global_pp()
        self.cube.data = self.cube.data - 260

    def test_exponentiate(self):
        a = self.cube
        a.data = a.data.astype(np.float64)
        e = pow(a, 4)
        self.assertCMLApproxData(e, ('analysis', 'exponentiate.cml'))

    def test_square_root(self):
        # Make sure we have something which we can take the root of.
        a = self.cube
        a.data = abs(a.data)
        a.units **= 2

        e = a ** 0.5

        self.assertCML(e, ('analysis', 'sqrt.cml'))
        self.assertArrayEqual(e.data, a.data ** 0.5)
        self.assertRaises(ValueError, iris.analysis.maths.exponentiate, a, 0.3)


class TestExponential(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.simple_1d()

    def test_exp(self):
        e = iris.analysis.maths.exp(self.cube)
        self.assertCMLApproxData(e, ('analysis', 'exp.cml'))


@iris.tests.skip_data
class TestLog(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.global_pp()

    def test_log(self):
        e = iris.analysis.maths.log(self.cube)
        self.assertCMLApproxData(e, ('analysis', 'log.cml'))

    def test_log2(self):
        e = iris.analysis.maths.log2(self.cube)
        self.assertCMLApproxData(e, ('analysis', 'log2.cml'))

    def test_log10(self):
        e = iris.analysis.maths.log10(self.cube)
        self.assertCMLApproxData(e, ('analysis', 'log10.cml'))


class TestMaskedArrays(tests.IrisTest):
    ops = (operator.add, operator.sub, operator.mul, operator.div)
    iops = (operator.iadd, operator.isub, operator.imul, operator.idiv)

    def setUp(self):
        self.data1 = ma.MaskedArray([[9,9,9],[8,8,8,]],mask=[[0,1,0],[0,0,1]])
        self.data2 = ma.MaskedArray([[3,3,3],[2,2,2,]],mask=[[0,1,0],[0,1,1]])
        
        self.cube1 = iris.cube.Cube(self.data1) 
        self.cube2 = iris.cube.Cube(self.data2) 

    def test_operator(self):
        for test_op in self.ops:
            result1 = test_op(self.cube1, self.cube2)
            result2 = test_op(self.data1, self.data2)
        
            np.testing.assert_array_equal(result1.data, result2)

    def test_operator_in_place(self):
        for test_op in self.iops:
            test_op(self.cube1, self.cube2)
            test_op(self.data1, self.data2)

            np.testing.assert_array_equal(self.cube1.data, self.data1)
    
    def test_operator_scalar(self):
        for test_op in self.ops:
            result1 = test_op(self.cube1, 2)
            result2 = test_op(self.data1, 2)
        
            np.testing.assert_array_equal(result1.data, result2)

    def test_operator_array(self):
        for test_op in self.ops:
            result1 = test_op(self.cube1, self.data2)
            result2 = test_op(self.data1, self.data2)
    
            np.testing.assert_array_equal(result1.data, result2)

    def test_incompatible_dimensions(self):
        data3 = ma.MaskedArray([[3,3,3,4],[2,2,2]],mask=[[0,1,0,0],[0,1,1]])
        with self.assertRaises(ValueError):
            # incompatible dimensions
            self.cube1 + data3

    def test_increase_cube_dimensionality(self):
        with self.assertRaises(ValueError):
            # This would increase the dimensionality of the cube due to auto broadcasting
            cubex = iris.cube.Cube(ma.MaskedArray([[9,]],mask=[[0]])) 
            cubex + ma.MaskedArray([[3,3,3,3]],mask=[[0,1,0,1]]) 
            
    
if __name__ == "__main__":
    tests.main()
