# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test the iteration of cubes in step.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import itertools
import operator
import random
import warnings

import numpy as np

import iris
import iris.analysis
import iris.iterate
import iris.tests.stock
from functools import reduce


@tests.skip_data
class TestIterateFunctions(tests.IrisTest):
    def setUp(self):
        self.cube_a = iris.tests.stock.realistic_4d()[0, 0:5, 0:10, 0:12]
        self.cube_b = iris.tests.stock.realistic_4d()[1, 0:5, 0:10, 0:12]
        self.coord_names = ["grid_latitude", "grid_longitude"]

        # Modify elements of cube_b to introduce additional differences
        self.cube_b.attributes["source"] = "Iris iterate test case"
        self.cube_b.add_aux_coord(iris.coords.AuxCoord(23, long_name="other"))

    def test_izip_no_args(self):
        with self.assertRaises(TypeError):
            iris.iterate.izip()
        with self.assertRaises(TypeError):
            iris.iterate.izip(coords=self.coord_names)
        with self.assertRaises(TypeError):
            iris.iterate.izip(coords=self.coord_names, ordered=False)

    def test_izip_input_collections(self):
        # Should work with one or more cubes as args
        iris.iterate.izip(self.cube_a, coords=self.coord_names)
        iris.iterate.izip(self.cube_a, self.cube_a, coords=self.coord_names)
        iris.iterate.izip(self.cube_a, self.cube_b, coords=self.coord_names)
        iris.iterate.izip(
            self.cube_a, self.cube_b, self.cube_a, coords=self.coord_names
        )
        # Check unpacked collections
        cubes = [self.cube_a] * 10
        iris.iterate.izip(*cubes, coords=self.coord_names)
        cubes = tuple(cubes)
        iris.iterate.izip(*cubes, coords=self.coord_names)

    def test_izip_returns_iterable(self):
        try:
            # Raises an exception if arg is not iterable
            iter(iris.iterate.izip(self.cube_a, coords=self.coord_names))
        except TypeError:
            self.fail("iris.iterate.izip is not returning an iterable")

    def test_izip_unequal_slice_coords(self):
        # Create a cube with grid_latitude and grid_longitude coords
        # that differ in size from cube_a's
        other_cube = self.cube_a[0, 0:3, 0:3]
        nslices = self.cube_a.shape[0]
        i = 0
        for slice_a, slice_other in iris.iterate.izip(
            self.cube_a, other_cube, coords=self.coord_names
        ):
            slice_a_truth = self.cube_a[i, :, :]
            slice_other_truth = other_cube
            self.assertEqual(slice_a_truth, slice_a)
            self.assertEqual(slice_other_truth, slice_other)
            i += 1
        self.assertEqual(i, nslices)
        # Attempting to iterate over these incompatible coords should
        # raise an exception
        with self.assertRaises(ValueError):
            iris.iterate.izip(self.cube_a, other_cube)

    def test_izip_missing_slice_coords(self):
        # Remove latitude coordinate from one of the cubes
        other_cube = self.cube_b.copy()
        other_cube.remove_coord("grid_latitude")
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            iris.iterate.izip(self.cube_a, other_cube, coords=self.coord_names)
        # Create a cube with latitude and longitude rather than grid_latitude
        # and grid_longitude
        self.cube_b.coord("grid_latitude").rename("latitude")
        self.cube_b.coord("grid_longitude").rename("longitude")
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            iris.iterate.izip(
                self.cube_a, self.cube_b, coords=self.coord_names
            )

    def test_izip_onecube_no_coords(self):
        # Should do the same as slices() but bearing in mind izip.next()
        # returns a tuple of cubes

        # Reduce the size to speed things up - we're going to iterate
        # over *every* data point.
        self.cube_b = self.cube_b[:2, :4, :3]

        # Empty list as coords
        slice_iterator = self.cube_b.slices([])
        zip_iterator = iris.iterate.izip(self.cube_b, coords=[])
        for cube_slice in slice_iterator:
            # First element of tuple: (extractedcube, )
            zip_slice = next(zip_iterator)[0]
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            next(zip_iterator)  # Should raise exception if we continue try to
            # to iterate

    def test_izip_onecube_lat_lon(self):
        # Two coords
        slice_iterator = self.cube_b.slices(self.coord_names)
        zip_iterator = iris.iterate.izip(self.cube_b, coords=self.coord_names)
        for cube_slice in slice_iterator:
            # First element of tuple: (extractedcube, )
            zip_slice = next(zip_iterator)[0]
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            next(zip_iterator)  # Should raise exception if we continue to try
            # to iterate

    def test_izip_onecube_lat(self):
        # One coord
        slice_iterator = self.cube_b.slices("grid_latitude")
        zip_iterator = iris.iterate.izip(self.cube_b, coords="grid_latitude")
        for cube_slice in slice_iterator:
            # First element of tuple: (extractedcube, )
            zip_slice = next(zip_iterator)[0]
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            next(zip_iterator)  # Should raise exception if we continue to try
            # to iterate

    def test_izip_onecube_height_lat_long(self):
        # All coords
        slice_iterator = self.cube_b.slices(
            ["level_height", "grid_latitude", "grid_longitude"]
        )
        zip_iterator = iris.iterate.izip(
            self.cube_b,
            coords=["level_height", "grid_latitude", "grid_longitude"],
        )
        for cube_slice in slice_iterator:
            # First element of tuple: (extractedcube, )
            zip_slice = next(zip_iterator)[0]
            self.assertEqual(cube_slice, zip_slice)
        with self.assertRaises(StopIteration):
            next(zip_iterator)  # Should raise exception if we continue to try
            # to iterate

    def test_izip_same_cube_lat_lon(self):
        nslices = self.cube_b.shape[0]
        slice_iterator = self.cube_b.slices(self.coord_names)
        count = 0
        for slice_first, slice_second in iris.iterate.izip(
            self.cube_b, self.cube_b, coords=self.coord_names
        ):
            self.assertEqual(slice_first, slice_second)  # Equal to each other
            self.assertEqual(
                slice_first, next(slice_iterator)
            )  # Equal to the truth (from slice())
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_same_cube_lat(self):
        nslices = (
            self.cube_a.shape[0] * self.cube_a.shape[2]
        )  # Calc product of dimensions
        # excluding the latitude
        # (2nd data dim)
        slice_iterator = self.cube_a.slices("grid_latitude")
        count = 0
        for slice_first, slice_second in iris.iterate.izip(
            self.cube_a, self.cube_a, coords=["grid_latitude"]
        ):
            self.assertEqual(slice_first, slice_second)
            self.assertEqual(
                slice_first, next(slice_iterator)
            )  # Equal to the truth (from slice())
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_same_cube_no_coords(self):
        # Reduce the size to speed things up - we're going to iterate
        # over *every* data point.
        self.cube_b = self.cube_b[:2, :4, :3]

        nslices = reduce(operator.mul, self.cube_b.shape)
        slice_iterator = self.cube_b.slices([])
        count = 0
        for slice_first, slice_second in iris.iterate.izip(
            self.cube_b, self.cube_b, coords=[]
        ):
            self.assertEqual(slice_first, slice_second)
            self.assertEqual(
                slice_first, next(slice_iterator)
            )  # Equal to the truth (from slice())
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_subcube_of_same(self):
        for _ in range(3):
            super_cube = self.cube_a
            # Random int to pick coord value to calc subcube
            k = random.randint(0, super_cube.shape[0] - 1)
            sub_cube = super_cube[k, :, :]
            super_slice_iterator = super_cube.slices(self.coord_names)
            j = 0
            for super_slice, sub_slice in iris.iterate.izip(
                super_cube, sub_cube, coords=self.coord_names
            ):
                self.assertEqual(
                    sub_slice, sub_cube
                )  # This cube should not change
                # as lat and long are the only
                # data dimensions in this cube)
                self.assertEqual(super_slice, next(super_slice_iterator))
                if j == k:
                    self.assertEqual(super_slice, sub_slice)
                else:
                    self.assertNotEqual(super_slice, sub_slice)
                j += 1
            nslices = super_cube.shape[0]
            self.assertEqual(j, nslices)

    def test_izip_same_dims(self):
        # Check single coords slice
        nslices = reduce(operator.mul, self.cube_a.shape[1:])
        nslices_to_check = (
            20  # This is only approximate as we use random to select slices
        )
        # Fraction of slices to check
        check_eq_probability = max(0.0, min(1.0, nslices_to_check / nslices))

        ij_iterator = np.ndindex(self.cube_a.shape[1], self.cube_a.shape[2])
        count = 0
        for slice_a, slice_b in iris.iterate.izip(
            self.cube_a, self.cube_b, coords="level_height"
        ):
            i, j = next(ij_iterator)
            if random.random() < check_eq_probability:  # Check these slices
                slice_a_truth = self.cube_a[:, i, j]
                slice_b_truth = self.cube_b[:, i, j]
                self.assertEqual(slice_a_truth, slice_a)
                self.assertEqual(slice_b_truth, slice_b)
            count += 1
        self.assertEqual(count, nslices)
        # Two coords
        nslices = self.cube_a.shape[0]
        i_iterator = iter(range(self.cube_a.shape[0]))
        count = 0
        for slice_a, slice_b in iris.iterate.izip(
            self.cube_a, self.cube_b, coords=self.coord_names
        ):
            i = next(i_iterator)
            slice_a_truth = self.cube_a[i, :, :]
            slice_b_truth = self.cube_b[i, :, :]
            self.assertEqual(slice_a_truth, slice_a)
            self.assertEqual(slice_b_truth, slice_b)
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_extra_dim(self):
        big_cube = self.cube_a
        # Remove first data dimension and associated coords
        little_cube = self.cube_b.copy()
        for factory in little_cube.aux_factories:
            little_cube.remove_aux_factory(factory)
        little_cube = little_cube[0]
        little_cube.remove_coord("model_level_number")
        little_cube.remove_coord("level_height")
        little_cube.remove_coord("sigma")
        # little_slice should remain the same as there are no other data dimensions
        little_slice_truth = little_cube
        i = 0
        for big_slice, little_slice in iris.iterate.izip(
            big_cube, little_cube, coords=self.coord_names
        ):
            big_slice_truth = big_cube[i, :, :]
            self.assertEqual(little_slice_truth, little_slice)
            self.assertEqual(big_slice_truth, big_slice)
            i += 1
        nslices = big_cube.shape[0]
        self.assertEqual(nslices, i)

        # Leave middle coord but move it from a data dimension to a scalar coord by slicing
        little_cube = self.cube_b[:, 0, :]

        # Now remove associated coord
        little_cube.remove_coord("grid_latitude")
        # Check we raise an exception if we request coords one of the cubes doesn't have
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            iris.iterate.izip(big_cube, little_cube, coords=self.coord_names)

        # little_slice should remain the same as there are no other data dimensions
        little_slice_truth = little_cube
        i = 0
        for big_slice, little_slice in iris.iterate.izip(
            big_cube,
            little_cube,
            coords=["model_level_number", "grid_longitude"],
        ):
            big_slice_truth = big_cube[:, i, :]
            self.assertEqual(little_slice_truth, little_slice)
            self.assertEqual(big_slice_truth, big_slice)
            i += 1
        nslices = big_cube.shape[1]
        self.assertEqual(nslices, i)

        # Take a random slice reducing it to a 1d cube
        p = random.randint(0, self.cube_b.shape[0] - 1)
        q = random.randint(0, self.cube_b.shape[2] - 1)
        little_cube = self.cube_b[p, :, q]
        nslices = big_cube.shape[0] * big_cube.shape[2]
        nslices_to_check = (
            20  # This is only approximate as we use random to select slices
        )
        # Fraction of slices to check
        check_eq_probability = max(0.0, min(1.0, nslices_to_check / nslices))
        ij_iterator = np.ndindex(big_cube.shape[0], big_cube.shape[2])
        count = 0
        for big_slice, little_slice in iris.iterate.izip(
            big_cube, little_cube, coords="grid_latitude"
        ):
            i, j = next(ij_iterator)
            if random.random() < check_eq_probability:
                big_slice_truth = big_cube[i, :, j]
                little_slice_truth = (
                    little_cube  # Just 1d so slice is entire cube
                )
                self.assertEqual(little_slice_truth, little_slice)
                self.assertEqual(big_slice_truth, big_slice)
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_different_shaped_coords(self):
        other = self.cube_b[0:-1]
        # Different 'z' coord shape - expect a ValueError
        with self.assertRaises(ValueError):
            iris.iterate.izip(self.cube_a, other, coords=self.coord_names)

    def test_izip_different_valued_coords(self):
        # Change a value in one of the coord points arrays so they are no longer identical
        new_points = self.cube_b.coord("model_level_number").points.copy()
        new_points[0] = 0
        self.cube_b.coord("model_level_number").points = new_points
        # slice coords
        latitude = self.cube_b.coord("grid_latitude")
        longitude = self.cube_b.coord("grid_longitude")
        # Same coord metadata and shape, but different values - check it produces a warning
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error"
            )  # Cause all warnings to raise Exceptions
            with self.assertRaises(UserWarning):
                iris.iterate.izip(
                    self.cube_a, self.cube_b, coords=self.coord_names
                )
            # Call with coordinates, rather than names
            with self.assertRaises(UserWarning):
                iris.iterate.izip(
                    self.cube_a, self.cube_b, coords=[latitude, longitude]
                )
        # Check it still iterates through as expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nslices = self.cube_a.shape[0]
            i = 0
            for slice_a, slice_b in iris.iterate.izip(
                self.cube_a, self.cube_b, coords=self.coord_names
            ):
                slice_a_truth = self.cube_a[i, :, :]
                slice_b_truth = self.cube_b[i, :, :]
                self.assertEqual(slice_a_truth, slice_a)
                self.assertEqual(slice_b_truth, slice_b)
                self.assertNotEqual(slice_b, None)
                i += 1
            self.assertEqual(i, nslices)
            # Call with coordinate instances rather than coord names
            i = 0
            for slice_a, slice_b in iris.iterate.izip(
                self.cube_a, self.cube_b, coords=[latitude, longitude]
            ):
                slice_a_truth = self.cube_a[i, :, :]
                slice_b_truth = self.cube_b[i, :, :]
                self.assertEqual(slice_a_truth, slice_a)
                self.assertEqual(slice_b_truth, slice_b)
                i += 1
            self.assertEqual(i, nslices)

    def test_izip_ordered(self):
        # Remove coordinate that spans grid_latitude and
        # grid_longitude dimensions as this will be common between
        # the resulting cubes but differ in shape
        self.cube_b.remove_coord("surface_altitude")
        cube = self.cube_b.copy()
        cube.transpose([0, 2, 1])  # switch order of lat and lon
        nslices = self.cube_b.shape[0]
        # Default behaviour: ordered = True
        i = 0
        for slice_b, cube_slice in iris.iterate.izip(
            self.cube_b, cube, coords=self.coord_names, ordered=True
        ):
            slice_b_truth = self.cube_b[i, :, :]
            cube_slice_truth = cube[i, :, :]
            # izip should transpose the slice to ensure order is [lat, lon]
            cube_slice_truth.transpose()
            self.assertEqual(slice_b_truth, slice_b)
            self.assertEqual(cube_slice_truth, cube_slice)
            i += 1
        self.assertEqual(i, nslices)
        # Alternative behaviour: ordered=False (retain original ordering)
        i = 0
        for slice_b, cube_slice in iris.iterate.izip(
            self.cube_b, cube, coords=self.coord_names, ordered=False
        ):
            slice_b_truth = self.cube_b[i, :, :]
            cube_slice_truth = cube[i, :, :]
            self.assertEqual(slice_b_truth, slice_b)
            self.assertEqual(cube_slice_truth, cube_slice)
            i += 1
        self.assertEqual(i, nslices)

    def test_izip_use_in_analysis(self):
        # Calculate mean, collapsing vertical dimension
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vertical_mean = self.cube_b.collapsed(
                "model_level_number", iris.analysis.MEAN
            )
        nslices = self.cube_b.shape[0]
        i = 0
        for slice_b, mean_slice in iris.iterate.izip(
            self.cube_b, vertical_mean, coords=self.coord_names
        ):
            slice_b_truth = self.cube_b[i, :, :]
            self.assertEqual(slice_b_truth, slice_b)
            # Should return same cube in each iteration
            self.assertEqual(vertical_mean, mean_slice)
            i += 1
        self.assertEqual(i, nslices)

    def test_izip_nd_non_ortho(self):
        cube1 = iris.cube.Cube(np.zeros((5, 5, 5)))
        cube1.add_aux_coord(
            iris.coords.AuxCoord(np.arange(5), long_name="z"), [0]
        )
        cube1.add_aux_coord(
            iris.coords.AuxCoord(np.arange(25).reshape(5, 5), long_name="y"),
            [1, 2],
        )
        cube1.add_aux_coord(
            iris.coords.AuxCoord(np.arange(25).reshape(5, 5), long_name="x"),
            [1, 2],
        )
        cube2 = cube1.copy()

        # The two coords are not orthogonal so we cannot use them with izip
        with self.assertRaises(ValueError):
            iris.iterate.izip(cube1, cube2, coords=["y", "x"])

    def test_izip_nd_ortho(self):
        cube1 = iris.cube.Cube(np.zeros((5, 5, 5, 5, 5), dtype="f8"))
        cube1.add_dim_coord(
            iris.coords.DimCoord(
                np.arange(5, dtype="i8"), long_name="z", units="1"
            ),
            [0],
        )
        cube1.add_aux_coord(
            iris.coords.AuxCoord(
                np.arange(25, dtype="i8").reshape(5, 5),
                long_name="y",
                units="1",
            ),
            [1, 2],
        )
        cube1.add_aux_coord(
            iris.coords.AuxCoord(
                np.arange(25, dtype="i8").reshape(5, 5),
                long_name="x",
                units="1",
            ),
            [3, 4],
        )
        cube2 = cube1.copy()

        # The two coords are orthogonal so we can use them with izip
        it = iris.iterate.izip(cube1, cube2, coords=["y", "x"])
        cubes = list(np.array(list(it)).flatten())
        self.assertCML(cubes, ("iterate", "izip_nd_ortho.cml"))

    def _check_2d_slices(self):
        # Helper method to verify slices from izip match those from
        # cube.slices().
        slice_a_iterator = self.cube_a.slices(self.coord_names)
        slice_b_iterator = self.cube_b.slices(self.coord_names)
        nslices = self.cube_b.shape[0]
        count = 0
        for slice_a, slice_b in iris.iterate.izip(
            self.cube_a, self.cube_b, coords=self.coord_names
        ):
            self.assertEqual(slice_a, next(slice_a_iterator))
            self.assertEqual(slice_b, next(slice_b_iterator))
            count += 1
        self.assertEqual(count, nslices)

    def test_izip_extra_coords_step_dim(self):
        # Add extra different coords to cubes along the dimension we are
        # stepping through.
        coord_a = iris.coords.AuxCoord(
            np.arange(self.cube_a.shape[0]), long_name="another on a"
        )
        self.cube_a.add_aux_coord(coord_a, 0)
        coord_b = iris.coords.AuxCoord(
            np.arange(self.cube_b.shape[0]), long_name="another on b"
        )
        self.cube_b.add_aux_coord(coord_b, 0)
        # Check slices.
        self._check_2d_slices()

    def test_izip_extra_coords_slice_dim(self):
        # Add extra different coords to cubes along a dimension we are
        # not stepping through.
        coord_a = iris.coords.AuxCoord(
            np.arange(self.cube_a.shape[1]), long_name="another on a"
        )
        self.cube_a.add_aux_coord(coord_a, 1)
        coord_b = iris.coords.AuxCoord(
            np.arange(self.cube_b.shape[1]), long_name="another on b"
        )
        self.cube_b.add_aux_coord(coord_b, 1)
        self._check_2d_slices()

    def test_izip_extra_coords_both_slice_dims(self):
        # Add extra different coords to cubes along the dimensions we are
        # not stepping through.
        coord_a = iris.coords.AuxCoord(
            np.arange(self.cube_a.shape[1]), long_name="another on a"
        )
        self.cube_a.add_aux_coord(coord_a, 1)
        coord_b = iris.coords.AuxCoord(
            np.arange(self.cube_b.shape[2]), long_name="another on b"
        )
        self.cube_b.add_aux_coord(coord_b, 2)
        self._check_2d_slices()

    def test_izip_no_common_coords_on_step_dim(self):
        # Change metadata on all coords along the dimension we are
        # stepping through.
        self.cube_a.coord("model_level_number").rename("foo")
        self.cube_a.coord("sigma").rename("bar")
        self.cube_a.coord("level_height").rename("woof")
        # izip should step through them as a product.
        slice_a_iterator = self.cube_a.slices(self.coord_names)
        slice_b_iterator = self.cube_b.slices(self.coord_names)
        product_iterator = itertools.product(
            slice_a_iterator, slice_b_iterator
        )
        nslices = self.cube_a.shape[0] * self.cube_b.shape[0]
        count = 0
        for slice_a, slice_b in iris.iterate.izip(
            self.cube_a, self.cube_b, coords=self.coord_names
        ):
            expected_a, expected_b = next(product_iterator)
            self.assertEqual(slice_a, expected_a)
            self.assertEqual(slice_b, expected_b)
            count += 1
        self.assertEqual(count, nslices)


if __name__ == "__main__":
    tests.main()
