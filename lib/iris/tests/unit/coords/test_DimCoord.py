# (C) British Crown Copyright 2017, Met Office
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
Unit tests for the :class:`iris.coords.DimCoord` class.

Note: a lot of these methods are actually defined by the :class:`Coord` class,
but can only be tested on concrete DimCoord/AuxCoord instances.
In addition, the DimCoord class has little behaviour for some of these, as it
cannot contain lazy points or bounds data.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import mock
import numpy as np
import unittest

from iris._lazy_data import is_lazy_data

from iris.coords import DimCoord


def arrays_share_data(a1, a2):
    # Check whether 2 real arrays with the same content view the same data.
    # Notes:
    # *  requires that array contents are initially identical
    # *  forces a1 to be writeable and modifies it
    assert np.all(a1 == a2)
    a1.flags.writeable = True
    a1 += np.array(1.0, dtype=a1.dtype)
    return np.all(a1 == a2)


def setup_test_arrays(self, shape=(3,)):
    n_pts = np.prod(shape)
    # Note: values need to be ints for testing integer dtypes.
    points = 10.0 * np.arange(n_pts, dtype=float).reshape(shape)
    lower = points - 2.0
    upper = points + 2.0
    bounds = np.stack((lower, upper), axis=-1)
    self.pts_real = points
    self.pts_lazy = da.from_array(points, points.shape)
    self.bds_real = bounds
    self.bds_lazy = da.from_array(bounds, bounds.shape)


class Test__init__(tests.IrisTest):
    # Test for DimCoord creation, with various combinations of points and
    # bounds = real / lazy / None.
    def setUp(self):
        setup_test_arrays(self)

    def test_real_points(self):
        # Check that coord creation does not copy real points data, but makes
        # a readonly view of it.
        coord = DimCoord(self.pts_real)
        pts = coord.core_points()
        self.assertIsInstance(pts, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertTrue(arrays_share_data(pts, self.pts_real),
                        'Points are not the same data as the provided array.')

    def test_lazy_points(self):
        # Check that coord creation realises lazy points data.
        coord = DimCoord(self.pts_lazy)
        pts = coord.core_points()
        self.assertIsInstance(pts, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)

    def test_real_points_with_real_bounds(self):
        # Check that coord creation does not copy real bounds data, but makes
        # a readonly view of it.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, np.ndarray)
        self.assertIsInstance(bds, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertArrayEqual(bds, self.bds_real)
        self.assertTrue(arrays_share_data(pts, self.pts_real),
                        'Points are not the same data as the provided array.')
        self.assertTrue(arrays_share_data(bds, self.bds_real),
                        'Bounds are not the same data as the provided array.')

    def test_real_points_with_lazy_bounds(self):
        ' Check that coord creation realises lazy bounds.'
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, np.ndarray)
        self.assertIsInstance(bds, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertArrayEqual(bds, self.bds_real)
        self.assertTrue(arrays_share_data(pts, self.pts_real),
                        'Points are not the same data as the provided array.')

    def test_lazy_points_with_real_bounds(self):
        coord = DimCoord(self.pts_lazy, bounds=self.bds_real)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, np.ndarray)
        self.assertIsInstance(bds, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertArrayEqual(bds, self.bds_real)
        self.assertTrue(arrays_share_data(bds, self.bds_real),
                        'Bounds are not the same data as the provided array.')

    def test_lazy_points_with_lazy_bounds(self):
        coord = DimCoord(self.pts_lazy, bounds=self.bds_lazy)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, np.ndarray)
        self.assertIsInstance(bds, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertArrayEqual(bds, self.bds_real)

    def test_fail_bounds_shape_mismatch(self):
        bds_shape = list(self.bds_real.shape)
        bds_shape[0] += 1
        bds_wrong = np.zeros(bds_shape)
        msg = 'shape'
        with self.assertRaisesRegexp(ValueError, msg):
            DimCoord(self.pts_real, bounds=bds_wrong)

    def test_fail_nonmonotonic(self):
        msg = 'must be strictly monotonic'
        with self.assertRaisesRegexp(ValueError, msg):
            DimCoord([1, 2, 0, 3])


class Test_core_points(tests.IrisTest):
    # Test for DimCoord.core_points() with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_real_points(self):
        data = self.pts_real
        coord = DimCoord(data)
        result = coord.core_points()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.pts_real)
        self.assertTrue(
            arrays_share_data(result, self.pts_real),
            'core_points() are not the same data as the internal array.')

    def test_lazy_points(self):
        lazy_data = self.pts_lazy
        coord = DimCoord(lazy_data)
        result = coord.core_points()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.pts_real)


class Test_core_bounds(tests.IrisTest):
    # Test for DimCoord.core_bounds() with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_no_bounds(self):
        coord = DimCoord(self.pts_real)
        result = coord.core_bounds()
        self.assertIsNone(result)

    def test_real_bounds(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.core_bounds()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.bds_real)
        self.assertTrue(
            arrays_share_data(result, self.bds_real),
            'core_bounds() are not the same data as the internal array.')

    def test_lazy_bounds(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.core_bounds()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.bds_real)


class Test_lazy_points(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_core(self):
        coord = DimCoord(self.pts_real)
        result = coord.lazy_points()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.dtype, self.pts_real.dtype)
        self.assertArrayEqual(result.compute(), self.pts_real)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_lazy)
        result = coord.lazy_points()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.dtype, self.pts_real.dtype)
        self.assertArrayEqual(result.compute(), self.pts_real)


class Test_lazy_bounds(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_no_bounds(self):
        coord = DimCoord(self.pts_real)
        result = coord.lazy_bounds()
        self.assertIsNone(result)

    def test_real_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.lazy_bounds()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.dtype, self.bds_real.dtype)
        self.assertArrayEqual(result.compute(), self.bds_real)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.lazy_bounds()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.dtype, self.bds_real.dtype)
        self.assertArrayEqual(result.compute(), self.bds_real)


class Test_has_lazy_points(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_core(self):
        coord = DimCoord(self.pts_real)
        result = coord.has_lazy_points()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_lazy)
        result = coord.has_lazy_points()
        self.assertFalse(result)


class Test_has_lazy_bounds(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.has_lazy_bounds()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.has_lazy_bounds()
        self.assertFalse(result)


class Test_bounds_dtype(tests.IrisTest):
    def test_i16(self):
        test_dtype = np.int16
        coord = DimCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertTrue(result == test_dtype)

    def test_u16(self):
        test_dtype = np.uint16
        coord = DimCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertTrue(result == test_dtype)

    def test_f16(self):
        test_dtype = np.float16
        coord = DimCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertTrue(result == test_dtype)


def lazyness_string(data):
    return 'lazy' if isinstance(data, da.Array) else 'real'


def coords_all_dtypes_and_lazynesses(self, dtypes=(np.float64, np.int16)):
    # Generate coords with all possible types of points and bounds, and all
    # of the given dtypes.
    points_types = ['real', 'lazy']
    bounds_types = ['no', 'real', 'lazy']
    for dtype in (np.float64, np.int16):
        for points_type_name in points_types:
            for bounds_type_name in bounds_types:
                pts = np.asarray(self.pts_real, dtype=dtype)
                bds = np.asarray(self.bds_real, dtype=dtype)
                if points_type_name == 'lazy':
                    pts = da.from_array(pts, pts.shape)
                if bounds_type_name == 'lazy':
                    bds = da.from_array(bds, bds.shape)
                elif bounds_type_name == 'no':
                    bds = None
                coord = DimCoord(pts, bounds=bds)
                result = (coord, points_type_name, bounds_type_name)
                yield result


class Test__getitem__(tests.IrisTest):
    # Test for DimCoord indexing with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_dtypes(self):
        # Index coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # Check that dtypes are preserved in all cases, taking dtypes directly
        # from the core points and bounds arrays (as we have no masking).
        for (main_coord, points_type_name, bounds_type_name) in \
                coords_all_dtypes_and_lazynesses(self):

            sub_coord = main_coord[:2]

            coord_dtype = main_coord.dtype
            msg = ('Indexing main_coord of dtype {} '
                   'with {} points and {} bounds '
                   'changed dtype of {} to {}.')

            sub_points = sub_coord.core_points()
            self.assertEqual(
                sub_points.dtype, coord_dtype,
                msg.format(coord_dtype,
                           points_type_name, bounds_type_name,
                           'points', sub_points.dtype))

            if bounds_type_name is not 'no':
                sub_bounds = sub_coord.core_bounds()
                self.assertEqual(
                    sub_bounds.dtype, coord_dtype,
                    msg.format(coord_dtype,
                               points_type_name, bounds_type_name,
                               'bounds', sub_points.dtype))

    def test_lazyness(self):
        # Index coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # Check that laziness is preserved in all cases.
        for (main_coord, points_type_name, bounds_type_name) in \
                coords_all_dtypes_and_lazynesses(self):
            # N.B. 'points_type_name' and 'bounds_type_name' in the iteration
            # are the original types (lazy/real/none) of the points+bounds,
            # but the DimCoord itself only ever has real data.
            if points_type_name == 'lazy':
                points_type_name = 'real'
            if bounds_type_name == 'lazy':
                bounds_type_name = 'real'

            sub_coord = main_coord[:2]

            msg = ('Indexing coord of dtype {} '
                   'with {} points and {} bounds '
                   'changed "lazyness" of {} from {!r} to {!r}.')
            coord_dtype = main_coord.dtype
            sub_points_lazyness = lazyness_string(sub_coord.core_points())
            self.assertEqual(
                sub_points_lazyness, points_type_name,
                msg.format(coord_dtype,
                           points_type_name, bounds_type_name,
                           'points', points_type_name, sub_points_lazyness))

            if bounds_type_name is not 'no':
                sub_bounds_lazy = lazyness_string(sub_coord.core_bounds())
                self.assertEqual(
                    sub_bounds_lazy, bounds_type_name,
                    msg.format(coord_dtype,
                               points_type_name, bounds_type_name,
                               'bounds', bounds_type_name, sub_bounds_lazy))

    def test_real_data_copies(self):
        # Index coords with all combinations of real+lazy points+bounds.
        # In all cases, check that any real arrays are copied by the indexing.
        for (main_coord, points_lazyness, bounds_lazyness) in \
                coords_all_dtypes_and_lazynesses(self, dtypes=[np.float32]):

            sub_coord = main_coord[:2]

            msg = ('Indexed coord with {} points and {} bounds '
                   'does not have its own separate {} array.')
            if points_lazyness == 'real':
                main_points = main_coord.core_points()
                sub_points = sub_coord.core_points()
                self.assertIsInstance(main_points, np.ndarray)
                self.assertIsInstance(sub_points, np.ndarray)
                self.assertArrayEqual(sub_points, main_points[:2])
                self.assertFalse(
                    arrays_share_data(sub_points, main_points[:2]),
                    msg.format(points_lazyness, bounds_lazyness, 'points'))

            if bounds_lazyness == 'real':
                main_bounds = main_coord.core_bounds()
                sub_bounds = sub_coord.core_bounds()
                self.assertIsInstance(main_bounds, np.ndarray)
                self.assertIsInstance(sub_bounds, np.ndarray)
                self.assertArrayEqual(sub_bounds, main_bounds[:2])
                self.assertFalse(
                    arrays_share_data(sub_bounds, main_bounds[:2]),
                    msg.format(points_lazyness, bounds_lazyness, 'bounds'))


class Test_copy(tests.IrisTest):
    # Test for DimCoord.copy() with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_writable_points(self):
        coord1 = DimCoord(np.arange(5),
                          bounds=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        coord2 = coord1.copy()
        msg = 'destination is read-only'

        with self.assertRaisesRegexp(ValueError, msg):
            coord1.points[:] = 0

        with self.assertRaisesRegexp(ValueError, msg):
            coord2.points[:] = 0

        with self.assertRaisesRegexp(ValueError, msg):
            coord1.bounds[:] = 0

        with self.assertRaisesRegexp(ValueError, msg):
            coord2.bounds[:] = 0

    def test_realdata_readonly(self):
        # Copy coords with all combinations of real+lazy points+bounds.
        # In all cases, check that data arrays are read-only.
        for (main_coord, points_type_name, bounds_type_name) in \
                coords_all_dtypes_and_lazynesses(self, dtypes=[np.float32]):

            copied_coord = main_coord.copy()

            msg = ('Copied coord with {} points and {} bounds '
                   'does not have read-only {}.')

            copied_points = copied_coord.core_points()
            with self.assertRaisesRegexp(ValueError, 'read-only',
                                         msg=msg.format(points_type_name,
                                                        bounds_type_name,
                                                        'points')):
                copied_points[:1] += 33

            if bounds_type_name != 'no':
                copied_bounds = copied_coord.core_bounds()
                with self.assertRaisesRegexp(ValueError, 'read-only',
                                             msg=msg.format(points_type_name,
                                                            bounds_type_name,
                                                            'bounds')):
                    copied_bounds[:1] += 33


class Test_points__getter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_points(self):
        # Getting real points does not change or copy them.
        coord = DimCoord(self.pts_real)
        result = coord.core_points()
        self.assertIs(result, self.pts_real,
                      'Points are not the same array as the provided data.')


class Test_points__setter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_set_real(self):
        # Setting points does not copy, but makes a readonly view.
        coord = DimCoord(self.pts_real)
        new_pts = self.pts_real + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertArrayEqual(result, new_pts)
        self.assertTrue(arrays_share_data(result, new_pts),
                        'Points are not the same data as the assigned array.')

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        coord = DimCoord([1.0, 2.0])
        msg = 'shape'
        with self.assertRaisesRegexp(ValueError, msg):
            coord.points = np.array([1.0, 2.0, 3.0])

    def test_set_lazy(self):
        # Setting new lazy points realises them.
        coord = DimCoord(self.pts_real)
        new_pts = self.pts_lazy + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, new_pts.compute())


class Test_bounds__getter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_bounds(self):
        # Getting real bounds does not change or copy them.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.bounds
        self.assertIs(result, self.bds_real,
                      'Points are not the same array as the provided data.')


class Test_bounds__setter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_set_real(self):
        # Setting bounds does not copy, but makes a readonly view.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        new_bounds = self.bds_real + 102.3
        coord.bounds = new_bounds
        result = coord.core_bounds()
        self.assertArrayEqual(result, new_bounds)
        self.assertTrue(arrays_share_data(result, new_bounds),
                        'Bounds are not the same data as the assigned array.')

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        msg = 'shape'
        with self.assertRaisesRegexp(ValueError, msg):
            coord.bounds = np.array([1.0, 2.0, 3.0])

    def test_set_lazy(self):
        # Setting new lazy bounds realises them.
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        new_bounds = self.bds_lazy + 102.3
        coord.bounds = new_bounds
        result = coord.core_bounds()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, new_bounds.compute())


if __name__ == '__main__':
    tests.main()
