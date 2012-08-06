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
Test the iteration of cubes in step.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import operator
import random
import warnings

import numpy

import iris
import iris.analysis
import iris.fileformats.pp
import iris.iterate


@iris.tests.skip_data
class TestIterateFunctions(tests.IrisTest):

    def setUp(self):
        self.airtemp, self.humidity = iris.load_strict(iris.tests.get_data_path(('PP', 'globClim1', 'dec_subset.pp')), 
                                                       ['air_potential_temperature', 'specific_humidity'])
        # Reduce the size of cubes to make tests run a bit quicker
        self.airtemp = self.airtemp[0:5, 0:10, 0:12]
        self.humidity = self.humidity[0:5, 0:10, 0:12]
        self.coords = ['latitude', 'longitude']
    
    def test_izip_no_args(self):
        with self.assertRaises(TypeError):
            iris.iterate.izip()
        with self.assertRaises(TypeError):
            iris.iterate.izip(coords=self.coords)
        with self.assertRaises(TypeError):
            iris.iterate.izip(coords=self.coords, ordered=False)

    def test_izip_input_collections(self):
        # Should work with one or more cubes as args
        iris.iterate.izip(self.airtemp, coords=self.coords)
        iris.iterate.izip(self.airtemp, self.airtemp, coords=self.coords)
        iris.iterate.izip(self.airtemp, self.humidity, coords=self.coords)
        iris.iterate.izip(self.airtemp, self.humidity, self.airtemp, coords=self.coords)
        # Check unpacked collections
        cubes = [self.airtemp]*10
        iris.iterate.izip(*cubes, coords=self.coords)
        cubes = tuple(cubes)
        iris.iterate.izip(*cubes, coords=self.coords)
    
    def test_izip_returns_iterable(self):
        try:
            iter(iris.iterate.izip(self.airtemp, coords=self.coords))    # Raises an exception if arg is not iterable
        except TypeError:
            self.fail('iris.iterate.izip is not returning an iterable')

    def test_izip_unequal_slice_coords(self):
        # Has latitude and longitude coords, but they differ from airtemp's
        other_cube = iris.load_strict(iris.tests.get_data_path(('PP', 'ocean_rle', 'ocean_rle.pp')), 
                                      iris.AttributeConstraint(STASH=iris.fileformats.pp.STASH(02, 30, 248)))
        nslices = self.airtemp.shape[0]
        i = 0
        for air_slice, cube_slice in iris.iterate.izip(self.airtemp, other_cube, coords=['latitude', 'longitude']):
            air_slice_truth = self.airtemp[i, :, :]
            cube_slice_truth = other_cube
            self.assertEqual(air_slice_truth, air_slice)
            self.assertEqual(cube_slice_truth, cube_slice)
            i += 1
        self.assertEqual(i, nslices)
        # Attempting to iterate over these incompatible coords should raise an exception
        with self.assertRaises(ValueError):
            iris.iterate.izip(self.airtemp, other_cube)
    
    def test_izip_missing_slice_coords(self):
        # Remove latitude coordinate from one of the cubes
        self.humidity.remove_coord('latitude')
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            iris.iterate.izip(self.airtemp, self.humidity, coords=self.coords)
        # Use a cube with grid_latitude and grid_longitude rather than latitude and longitude
        othercube = iris.load_strict(iris.tests.get_data_path(('PP', 'uk4', 'uk4par09.pp')), 'air_temperature')
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            iris.iterate.izip(self.airtemp, othercube, coords=self.coords)

    def test_izip_onecube(self):
        # Should do the same as slices() but bearing in mind izip.next() returns a tuple of cubes
        # Empty list as coords
        slice_iterator = self.humidity.slices([])
        zip_iterator = iris.iterate.izip(self.humidity, coords=[])
        for cube_slice in slice_iterator:
            zip_slice = zip_iterator.next()[0]    # First element of tuple: (extractedcube, )
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            zip_iterator.next()    # Should raise exception if we continue to try to iterate
        # Two coords
        slice_iterator = self.humidity.slices(['latitude', 'longitude'])
        zip_iterator = iris.iterate.izip(self.humidity, coords=['latitude', 'longitude'])
        for cube_slice in slice_iterator:
            zip_slice = zip_iterator.next()[0]    # First element of tuple: (extractedcube, )
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            zip_iterator.next()    # Should raise exception if we continue to try to iterate
        # One coord
        slice_iterator = self.humidity.slices('latitude')
        zip_iterator = iris.iterate.izip(self.humidity, coords='latitude')
        for cube_slice in slice_iterator:
            zip_slice = zip_iterator.next()[0]    # First element of tuple: (extractedcube, )
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            zip_iterator.next()    # Should raise exception if we continue to try to iterate
        # All coords
        slice_iterator = self.humidity.slices(['level_height', 'latitude', 'longitude'])
        zip_iterator = iris.iterate.izip(self.humidity, coords=['level_height', 'latitude', 'longitude'])
        for cube_slice in slice_iterator:
            zip_slice = zip_iterator.next()[0]    # First element of tuple: (extractedcube, )
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            zip_iterator.next()    # Should raise exception if we continue to try to iterate

    def test_izip_same_cube(self):
        nslices = self.humidity.shape[0]
        slice_iterator = self.humidity.slices(['latitude', 'longitude'])
        count = 0
        for slice_first, slice_second in iris.iterate.izip(self.humidity, self.humidity, coords=['latitude', 'longitude']):
            self.assertEqual(slice_first, slice_second)  # Equal to each other
            self.assertEqual(slice_first, slice_iterator.next()) # Equal to the truth (from slice())
            count += 1
        self.assertEqual(count, nslices)
        # Another case
        nslices = self.airtemp.shape[0] * self.airtemp.shape[2] # Calc product of dimensions excluding the latitude (2nd data dim)
        slice_iterator = self.airtemp.slices('latitude')
        count = 0
        for slice_first, slice_second in iris.iterate.izip(self.airtemp, self.airtemp, coords=['latitude']):
            self.assertEqual(slice_first, slice_second)
            self.assertEqual(slice_first, slice_iterator.next()) # Equal to the truth (from slice())
            count += 1
        self.assertEqual(count, nslices)
        # third case - full iteration
        nslices = reduce(operator.mul, self.humidity.shape)
        slice_iterator = self.humidity.slices([])
        count = 0
        for slice_first, slice_second in iris.iterate.izip(self.humidity, self.humidity, coords=[]):
            self.assertEqual(slice_first, slice_second)
            self.assertEqual(slice_first, slice_iterator.next()) # Equal to the truth (from slice())
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_subcube_of_same(self):
        for _ in xrange(3):
            super_cube = self.airtemp
            k = random.randint(0, super_cube.shape[0]-1)   # Random int to pick coord value to calc subcube
            sub_cube = super_cube[k, :, :]
            super_slice_iterator = super_cube.slices(['latitude', 'longitude'])
            j = 0
            for super_slice, sub_slice in iris.iterate.izip(super_cube, sub_cube , coords=['latitude', 'longitude']):
                self.assertEqual(sub_slice, sub_cube)    # This cube should not change as lat and long are 
                                                         # the only data dimensions in this cube)
                self.assertEqual(super_slice, super_slice_iterator.next())
                if j == k:
                    self.assertEqual(super_slice, sub_slice)
                else:
                    self.assertNotEqual(super_slice, sub_slice)
                j += 1
            nslices = super_cube.shape[0]
            self.assertEqual(j, nslices)

    def test_izip_same_dims(self):
        # Check single coords slice
        nslices = reduce(operator.mul, self.airtemp.shape[1:])
        nslices_to_check = 20       # This is only approximate as we use random to select slices
        check_eq_probability = max(0.0, min(1.0, float(nslices_to_check)/nslices))    # Fraction of slices to check
        ij_iterator = numpy.ndindex(self.airtemp.shape[1], self.airtemp.shape[2])
        count = 0
        for air_slice, hum_slice in iris.iterate.izip(self.airtemp, self.humidity, coords='level_height'):
            i, j = ij_iterator.next()
            if random.random() <  check_eq_probability:     # Check these slices
                air_slice_truth = self.airtemp[:, i, j]
                hum_slice_truth = self.humidity[:, i, j]
                self.assertEqual(air_slice_truth, air_slice)
                self.assertEqual(hum_slice_truth, hum_slice)
            count += 1
        self.assertEqual(count, nslices)
        # Two coords
        nslices = self.airtemp.shape[0]        
        i_iterator = iter(xrange(self.airtemp.shape[0]))
        count = 0
        for air_slice, hum_slice in iris.iterate.izip(self.airtemp, self.humidity, coords=['latitude', 'longitude']):
            i = i_iterator.next()
            air_slice_truth = self.airtemp[i, :, :]
            hum_slice_truth = self.humidity[i, :, :]
            self.assertEqual(air_slice_truth, air_slice)
            self.assertEqual(hum_slice_truth, hum_slice)
            count += 1
        self.assertEqual(count, nslices)
    
    def test_izip_extra_dim(self):
        big_cube = self.airtemp
        # Remove first data dimension and associated coords
        little_cube = self.humidity[0]  
        little_cube.remove_coord('model_level_number')
        little_cube.remove_coord('level_height')
        little_cube.remove_coord('sigma')
        # little_slice should remain the same as there are no other data dimensions 
        little_slice_truth = little_cube
        i = 0
        for big_slice, little_slice in iris.iterate.izip(big_cube, little_cube, coords=['latitude', 'longitude']):
            big_slice_truth = big_cube[i, :, :]
            self.assertEqual(little_slice_truth, little_slice)
            self.assertEqual(big_slice_truth, big_slice)
            i += 1
        nslices = big_cube.shape[0]
        self.assertEqual(nslices, i)
            
        # Leave middle coord but move it from a data dimension to a scalar coord by slicing
        little_cube = self.humidity[:, 0, :]

        # Now remove associated coord
        little_cube.remove_coord('latitude')
        # Check we raise an exception if we request coords one of the cubes doesn't have
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            iris.iterate.izip(big_cube, little_cube, coords=['latitude', 'longitude'])

        #little_slice should remain the same as there are no other data dimensions 
        little_slice_truth = little_cube
        i = 0
        for big_slice, little_slice in iris.iterate.izip(big_cube, little_cube, coords=['model_level_number', 'longitude']):
            big_slice_truth = big_cube[:, i, :]
            self.assertEqual(little_slice_truth, little_slice)
            self.assertEqual(big_slice_truth, big_slice)
            i += 1
        nslices = big_cube.shape[1]
        self.assertEqual(nslices, i)

        # Take a random slice reducing it to a 1d cube
        p = random.randint(0, self.humidity.shape[0]-1)
        q = random.randint(0, self.humidity.shape[2]-1)
        little_cube = self.humidity[p, :, q]         
        nslices = big_cube.shape[0]*big_cube.shape[2]
        nslices_to_check = 20   # This is only approximate as we use random to select slices
        check_eq_probability = max(0.0, min(1.0, float(nslices_to_check)/nslices))    # Fraction of slices to check
        ij_iterator = numpy.ndindex(big_cube.shape[0], big_cube.shape[2])
        count = 0
        for big_slice, little_slice in iris.iterate.izip(big_cube, little_cube, coords='latitude'):
            i, j = ij_iterator.next()
            if random.random() <  check_eq_probability:
                big_slice_truth = big_cube[i, :, j]
                little_slice_truth = little_cube    # Just 1d so slice is entire cube
                self.assertEqual(little_slice_truth, little_slice)
                self.assertEqual(big_slice_truth, big_slice)
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_different_shaped_coords(self): 
        other = self.humidity[0:-1]
        # Different 'z' coord shape - expect a ValueError
        with self.assertRaises(ValueError):
            iris.iterate.izip(self.airtemp, other, coords=['latitude', 'longitude'])

    def test_izip_different_valued_coords(self):
        # Change a value in one of the coord points arrays so they are no longer identical
        new_points = self.humidity.coord('model_level_number').points.copy()
        new_points[0] = 0
        self.humidity.coord('model_level_number').points = new_points
        # slice coords
        latitude = self.humidity.coord('latitude')
        longitude = self.humidity.coord('longitude')
        # Same coord metadata and shape, but different values - check it produces a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")    # Cause all warnings to raise Exceptions
            with self.assertRaises(UserWarning):
                iris.iterate.izip(self.airtemp, self.humidity, coords=['latitude', 'longitude'])
            # Call with coordinates, rather than names
            with self.assertRaises(UserWarning):
                iris.iterate.izip(self.airtemp, self.humidity, coords=[latitude, longitude])
        # Check it still iterates through as expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nslices = self.airtemp.shape[0]
            i = 0
            for air_slice, hum_slice in iris.iterate.izip(self.airtemp, self.humidity, coords=['latitude', 'longitude']):
                air_slice_truth = self.airtemp[i, :, :]
                hum_slice_truth = self.humidity[i, :, :]
                self.assertEqual(air_slice_truth, air_slice)
                self.assertEqual(hum_slice_truth, hum_slice)
                self.assertNotEqual(hum_slice, None)
                i += 1
            self.assertEqual(i, nslices)
            # Call with coordinate instances rather than coord names
            i = 0
            for air_slice, hum_slice in iris.iterate.izip(self.airtemp, self.humidity, coords=[latitude, longitude]):
                air_slice_truth = self.airtemp[i, :, :]
                hum_slice_truth = self.humidity[i, :, :]
                self.assertEqual(air_slice_truth, air_slice)
                self.assertEqual(hum_slice_truth, hum_slice)
                i += 1
            self.assertEqual(i, nslices)

    def test_izip_ordered(self):
        cube = self.humidity.copy()
        cube.transpose([0, 2, 1])     #switch order of lat and lon
        nslices = self.humidity.shape[0]        
        # Default behaviour: ordered = True 
        i = 0
        for hum_slice, cube_slice in iris.iterate.izip(self.humidity, cube, coords=['latitude', 'longitude'], ordered=True):
            hum_slice_truth = self.humidity[i, :, :]
            cube_slice_truth = cube[i, :, :]
            cube_slice_truth.transpose()    # izip should transpose the slice to ensure order is [lat, lon]
            self.assertEqual(hum_slice_truth, hum_slice)
            self.assertEqual(cube_slice_truth, cube_slice)
            i += 1
        self.assertEqual(i, nslices)
        # Alternative behaviour: ordered=False (retain original ordering)
        i = 0
        for hum_slice, cube_slice in iris.iterate.izip(self.humidity, cube, coords=['latitude', 'longitude'], ordered=False):
            hum_slice_truth = self.humidity[i, :, :]
            cube_slice_truth = cube[i, :, :]
            self.assertEqual(hum_slice_truth, hum_slice)
            self.assertEqual(cube_slice_truth, cube_slice)
            i += 1
        self.assertEqual(i, nslices)

    def test_izip_use_in_analysis(self):
        # Calculate mean, collapsing vertical dimension
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vertical_mean = self.humidity.collapsed('model_level_number', iris.analysis.MEAN)
        nslices = self.humidity.shape[0]
        i = 0
        for hum_slice, mean_slice in iris.iterate.izip(self.humidity, vertical_mean, coords=['latitude', 'longitude']):
            hum_slice_truth = self.humidity[i, :, :]
            self.assertEqual(hum_slice_truth, hum_slice)
            self.assertEqual(vertical_mean, mean_slice) # Should return same cube in each iteration
            i += 1
        self.assertEqual(i, nslices)
        
    def test_izip_nd_non_ortho(self):
        cube1 = iris.cube.Cube(numpy.zeros((5,5,5)))
        cube1.add_aux_coord(iris.coords.AuxCoord(numpy.arange(5), long_name="z"), [0])
        cube1.add_aux_coord(iris.coords.AuxCoord(numpy.arange(25).reshape(5,5), long_name="y"), [1,2])
        cube1.add_aux_coord(iris.coords.AuxCoord(numpy.arange(25).reshape(5,5), long_name="x"), [1,2])
        cube2 = cube1.copy()

        # The two coords are not orthogonal so we cannot use them with izip
        with self.assertRaises(ValueError):
            iris.iterate.izip(cube1, cube2, coords=["y", "x"])

    def test_izip_nd_ortho(self):
        cube1 = iris.cube.Cube(numpy.zeros((5,5,5,5,5)))
        cube1.add_dim_coord(iris.coords.AuxCoord(numpy.arange(5), long_name="z"), [0])
        cube1.add_aux_coord(iris.coords.AuxCoord(numpy.arange(25).reshape(5,5), long_name="y"), [1,2])
        cube1.add_aux_coord(iris.coords.AuxCoord(numpy.arange(25).reshape(5,5), long_name="x"), [3,4])
        cube2 = cube1.copy()
        
        # The two coords are orthogonal so we can use them with izip
        iter = iris.iterate.izip(cube1, cube2, coords=["y", "x"])
        cubes = list(numpy.array(list(iter)).flatten()) 
        self.assertCML(cubes, ('iterate', 'izip_nd_ortho.cml'))
        

if __name__ == '__main__':
    tests.main()

