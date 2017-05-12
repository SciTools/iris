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
Unit tests for the :class:`iris.coords.AuxCoord` class.

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

from iris.coords import AuxCoord


def arrays_share_data(a1, a2):
    # Check whether 2 real arrays with the same content view the same data.
    # Notes:
    # *  requires that array contents are initially identical
    # *  forces a1 to be writeable and modifies it
    assert np.all(a1 == a2)
    a1.flags.writeable = True
    a1 += np.array(1.0, dtype=a1.dtype)
    return np.all(a1 == a2)


def setup_test_arrays(self, shape=(2, 3)):
    n_pts = np.prod(shape)
    points = np.arange(n_pts, dtype=float).reshape(shape)
    lower = points - 0.2
    upper = points + 0.2
    bounds = np.stack((lower, upper), axis=-1)
    self.pts_real = points
    self.pts_lazy = da.from_array(points, points.shape)
    self.bds_real = bounds
    self.bds_lazy = da.from_array(bounds, bounds.shape)


class Test__init__(tests.IrisTest):
    # Test for AuxCoord creation, with various combinations of points and
    # bounds = real / lazy / None.
    def setUp(self):
        setup_test_arrays(self)

    def test_real_points(self):
        # Check coord creation does not copy real points data.
        coord = AuxCoord(self.pts_real)
        pts = coord.core_points()
        self.assertIsInstance(pts, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertTrue(arrays_share_data(pts, self.pts_real),
                        'Points do not share data with the provided array.')

    def test_lazy_points(self):
        coord = AuxCoord(self.pts_lazy)
        pts = coord.core_points()
        self.assertIsInstance(pts, da.Array)
        self.assertEqual(pts.dtype, self.pts_real.dtype)
        self.assertArrayEqual(pts.compute(), self.pts_real)

    def test_real_points_with_real_bounds(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, np.ndarray)
        self.assertIsInstance(bds, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertArrayEqual(bds, self.bds_real)
        self.assertTrue(arrays_share_data(pts, self.pts_real),
                        'Points do not share data with the provided array.')
        self.assertTrue(arrays_share_data(bds, self.bds_real),
                        'Bounds do not share data with the provided array.')

    def test_real_points_with_lazy_bounds(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, np.ndarray)
        self.assertArrayEqual(pts, self.pts_real)
        self.assertTrue(arrays_share_data(pts, self.pts_real),
                        'Points do not share data with the provided array.')
        self.assertIsInstance(bds, da.Array)
        self.assertEqual(bds.dtype, self.bds_real.dtype)
        self.assertArrayEqual(bds.compute(), self.bds_real)

    def test_lazy_points_with_real_bounds(self):
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_real)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, da.Array)
        self.assertEqual(pts.dtype, self.pts_real.dtype)
        self.assertArrayEqual(pts.compute(), self.pts_real)
        self.assertIsInstance(bds, np.ndarray)
        self.assertArrayEqual(bds, self.bds_real)
        self.assertTrue(arrays_share_data(bds, self.bds_real),
                        'Bounds do not share data with the provided array.')

    def test_lazy_points_with_lazy_bounds(self):
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy)
        pts = coord.core_points()
        bds = coord.core_bounds()
        self.assertIsInstance(pts, da.Array)
        self.assertEqual(pts.dtype, self.pts_real.dtype)
        self.assertArrayEqual(pts.compute(), self.pts_real)
        self.assertIsInstance(bds, da.Array)
        self.assertEqual(bds.dtype, self.bds_real.dtype)
        self.assertArrayEqual(bds.compute(), self.bds_real)

    def test_fail_bounds_shape_mismatch(self):
        bds_shape = list(self.bds_real.shape)
        bds_shape[0] += 1
        bds_wrong = np.zeros(bds_shape)
        msg = 'Bounds shape must be compatible with points shape'
        with self.assertRaisesRegexp(ValueError, msg):
            AuxCoord(self.pts_real, bounds=bds_wrong)


class Test_core_points(tests.IrisTest):
    # Test for AuxCoord.core_points() with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_real_points(self):
        coord = AuxCoord(self.pts_real)
        result = coord.core_points()
        self.assertArrayEqual(result, self.pts_real)
        self.assertTrue(
            arrays_share_data(result, self.pts_real),
            'core_points() do not share data with the internal array.')

    def test_lazy_points(self):
        lazy_data = self.pts_lazy
        coord = AuxCoord(lazy_data)
        result = coord.core_points()
        self.assertIs(result, lazy_data)

    def test_lazy_points_realise(self):
        coord = AuxCoord(self.pts_lazy)
        real_points = coord.points
        result = coord.core_points()
        self.assertIs(result, real_points)


class Test_core_bounds(tests.IrisTest):
    # Test for AuxCoord.core_bounds() with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_no_bounds(self):
        coord = AuxCoord(self.pts_real)
        result = coord.core_bounds()
        self.assertIsNone(result)

    def test_real_bounds(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.core_bounds()
        self.assertArrayEqual(result, self.bds_real)
        self.assertTrue(
            arrays_share_data(result, self.bds_real),
            'core_bounds() do not share data with the internal array.')

    def test_lazy_bounds(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.core_bounds()
        self.assertIsInstance(result, da.Array)
        self.assertEqual(result.dtype, self.bds_real.dtype)
        self.assertArrayEqual(result.compute(), self.bds_real)

    def test_lazy_bounds_realise(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        real_bounds = coord.bounds
        result = coord.core_bounds()
        self.assertIs(result, real_bounds)


class Test_lazy_points(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_core(self):
        coord = AuxCoord(self.pts_real)
        result = coord.lazy_points()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.dtype, self.pts_real.dtype)
        self.assertArrayEqual(result.compute(), self.pts_real)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_lazy)
        result = coord.lazy_points()
        self.assertIs(result, self.pts_lazy)


class Test_lazy_bounds(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_no_bounds(self):
        coord = AuxCoord(self.pts_real)
        result = coord.lazy_bounds()
        self.assertIsNone(result)

    def test_real_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.lazy_bounds()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.dtype, self.bds_real.dtype)
        self.assertArrayEqual(result.compute(), self.bds_real)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.lazy_bounds()
        self.assertIs(result, self.bds_lazy)

    def test_none(self):
        coord = AuxCoord(self.pts_real)
        result = coord.lazy_bounds()
        self.assertIsNone(result)


class Test_has_lazy_points(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_core(self):
        coord = AuxCoord(self.pts_real)
        result = coord.has_lazy_points()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_lazy)
        result = coord.has_lazy_points()
        self.assertTrue(result)

    def test_lazy_core_realise(self):
        coord = AuxCoord(self.pts_lazy)
        coord.points
        result = coord.has_lazy_points()
        self.assertFalse(result)


class Test_has_lazy_bounds(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.has_lazy_bounds()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.has_lazy_bounds()
        self.assertTrue(result)

    def test_lazy_core_realise(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        coord.bounds
        result = coord.has_lazy_bounds()
        self.assertFalse(result)


class Test_bounds_dtype(tests.IrisTest):
    def test_i16(self):
        test_dtype = np.int16
        coord = AuxCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertTrue(result == test_dtype)

    def test_u16(self):
        test_dtype = np.uint16
        coord = AuxCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertTrue(result == test_dtype)

    def test_f16(self):
        test_dtype = np.float16
        coord = AuxCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
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
                coord = AuxCoord(pts, bounds=bds)
                result = (coord, points_type_name, bounds_type_name)
                yield result


class Test__getitem__(tests.IrisTest):
    # Test for AuxCoord indexing with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_partial_slice_data_copy(self):
        parent_coord = AuxCoord([1., 2., 3.])
        sub_coord = parent_coord[:1]
        values_before_change = sub_coord.points.copy()
        parent_coord.points[:] = -999.9
        self.assertArrayEqual(sub_coord.points, values_before_change)

    def test_full_slice_data_copy(self):
        parent_coord = AuxCoord([1., 2., 3.])
        sub_coord = parent_coord[:]
        values_before_change = sub_coord.points.copy()
        parent_coord.points[:] = -999.9
        self.assertArrayEqual(sub_coord.points, values_before_change)

    def test_dtypes(self):
        # Index coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # Check that dtypes are preserved in all cases, taking dtypes directly
        # from the core points and bounds arrays (as we have no masking).
        for (main_coord, points_type_name, bounds_type_name) in \
                coords_all_dtypes_and_lazynesses(self):

            sub_coord = main_coord[:2, 1]

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

            sub_coord = main_coord[:2, 1]

            msg = ('Indexing coord of dtype {} '
                   'with {} points and {} bounds '
                   'changed laziness of {} from {!r} to {!r}.')
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

            sub_coord = main_coord[:2, 1]

            msg = ('Indexed coord with {} points and {} bounds '
                   'does not have its own separate {} array.')
            if points_lazyness == 'real':
                main_points = main_coord.core_points()
                sub_points = sub_coord.core_points()
                assert isinstance(main_points, np.ndarray)
                assert isinstance(sub_points, np.ndarray)
                assert np.all(sub_points == main_points[:2, 1])
                sub_points[:2] += 33
                linked = np.all(sub_points == main_points[:2, 1])
                self.assertFalse(
                    linked,
                    msg.format(points_lazyness, bounds_lazyness, 'points'))

            if bounds_lazyness == 'real':
                main_bounds = main_coord.core_bounds()
                sub_bounds = sub_coord.core_bounds()
                assert isinstance(main_bounds, np.ndarray)
                assert isinstance(sub_bounds, np.ndarray)
                assert np.all(sub_bounds == main_bounds[:2, 1])
                sub_bounds[:2] += 33
                linked = np.all(sub_bounds == main_bounds[:2, 1])
                self.assertFalse(
                    linked,
                    msg.format(points_lazyness, bounds_lazyness, 'bounds'))


class Test_copy(tests.IrisTest):
    # Test for AuxCoord.copy() with various types of points and bounds.
    def setUp(self):
        setup_test_arrays(self)

    def test_lazyness(self):
        # Copy coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # In all cases, check that real/lazy status is preserved..
        for (main_coord, points_lazyness, bounds_lazyness) in \
                coords_all_dtypes_and_lazynesses(self):

            coord_dtype = main_coord.dtype
            copied_coord = main_coord.copy()

            msg = ('Copying main_coord of dtype {} '
                   'with {} points and {} bounds '
                   'changed lazyness of {} from {!r} to {!r}.')

            copied_pts_lazyness = lazyness_string(copied_coord.core_points())
            self.assertEqual(copied_pts_lazyness, points_lazyness,
                             msg.format(coord_dtype,
                                        points_lazyness, bounds_lazyness,
                                        'points',
                                        points_lazyness, copied_pts_lazyness))

            if bounds_lazyness != 'no':
                copied_bds_lazy = lazyness_string(copied_coord.core_bounds())
                self.assertEqual(copied_bds_lazy, bounds_lazyness,
                                 msg.format(coord_dtype,
                                            points_lazyness, bounds_lazyness,
                                            'bounds',
                                            bounds_lazyness, copied_bds_lazy))

    def test_realdata_copies(self):
        # Copy coords with all combinations of real+lazy points+bounds.
        # In all cases, check that any real arrays are copies, not views.
        for (main_coord, points_lazyness, bounds_lazyness) in \
                coords_all_dtypes_and_lazynesses(self, dtypes=[np.float32]):

            copied_coord = main_coord.copy()

            msg = ('Copied coord with {} points and {} bounds '
                   'does not have its own separate {} array.')

            if points_lazyness == 'real':
                main_points = main_coord.core_points()
                copied_points = copied_coord.core_points()
                assert isinstance(main_points, np.ndarray)
                assert isinstance(copied_points, np.ndarray)
                assert np.all(main_points == copied_points)
                copied_points[:1, :1] += 33
                linked = np.all(main_points == copied_points)
                self.assertFalse(
                    linked,
                    msg.format(points_lazyness, bounds_lazyness, 'points'))

            if bounds_lazyness == 'real':
                main_bounds = main_coord.core_bounds()
                copied_bounds = copied_coord.core_bounds()
                assert isinstance(main_bounds, np.ndarray)
                assert isinstance(copied_bounds, np.ndarray)
                assert np.all(main_bounds == copied_bounds)
                copied_bounds[:1, :1] += 33
                linked = np.all(main_bounds == copied_bounds)
                self.assertFalse(
                    linked,
                    msg.format(points_lazyness, bounds_lazyness, 'bounds'))


class Test_points__getter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_mutable_real_points(self):
        # Check that coord.points returns a modifiable array, and changes to it
        # are reflected to the coord.
        data = np.array([1.0, 2.0, 3.0, 4.0])
        coord = AuxCoord(data)
        initial_values = data.copy()
        coord.points[1:2] += 33.1
        result = coord.points
        self.assertFalse(np.all(result == initial_values))

    def test_real_points(self):
        # Getting real points does not change or copy them.
        coord = AuxCoord(self.pts_real)
        result = coord.points
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.pts_real)
        self.assertTrue(arrays_share_data(result, self.pts_real),
                        'Points do not share data with the assigned array.')

    def test_lazy_points(self):
        # Getting lazy points realises them.
        coord = AuxCoord(self.pts_lazy)
        self.assertTrue(coord.has_lazy_points())
        result = coord.points
        self.assertFalse(coord.has_lazy_points())
        self.assertArrayEqual(result, self.pts_real)

    def test_real_points_with_real_bounds(self):
        # Getting real points does not change real bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        coord.points
        result = coord.core_bounds()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.bds_real)
        self.assertTrue(arrays_share_data(result, self.bds_real),
                        'Bounds do not share data with the assigned array.')

    def test_real_points_with_lazy_bounds(self):
        # Getting real points does not touch lazy bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        coord.points
        result = coord.core_bounds()
        self.assertIsInstance(result, da.Array)

    def test_lazy_points_with_real_bounds(self):
        # Getting lazy points does not affect real bounds.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_real)
        coord.points
        result = coord.core_bounds()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.bds_real)

    def test_lazy_points_with_lazy_bounds(self):
        # Getting lazy points does not touch lazy bounds.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy)
        coord.points
        result = coord.core_bounds()
        self.assertIsInstance(result, da.Array)


class Test_points__setter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_real_set_real(self):
        # Setting new real points does not make a copy.
        coord = AuxCoord(self.pts_real)
        new_pts = self.pts_real + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertArrayEqual(result, new_pts)
        self.assertTrue(arrays_share_data(result, new_pts))

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        coord = AuxCoord([1.0, 2.0])
        msg = 'shape'
        with self.assertRaisesRegexp(ValueError, msg):
            coord.points = np.array([1.0, 2.0, 3.0])

    def test_real_set_lazy(self):
        # Setting new lazy points does not make a copy.
        coord = AuxCoord(self.pts_real)
        new_pts = self.pts_lazy + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.compute(), new_pts.compute())

    def test_set_points_with_lazy_bounds(self):
        # Setting points does not touch lazy bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        new_pts = self.pts_real + 102.3
        coord.points = new_pts
        result = coord.core_bounds()
        self.assertIsInstance(result, da.Array)
        self.assertArrayEqual(result.compute(), self.bds_real)


class Test_bounds__getter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_mutable_real_bounds(self):
        # Check that coord.bounds returns a modifiable array, and changes to it
        # are reflected to the coord.
        pts_data = np.array([1.5, 2.5])
        bds_data = np.array([[1.4, 1.6], [2.4, 2.6]])
        coord = AuxCoord(pts_data, bounds=bds_data)
        initial_values = bds_data.copy()
        coord.bounds[1:2] += 33.1
        result = coord.bounds
        self.assertFalse(np.all(result == initial_values))

    def test_real_bounds(self):
        # Getting real bounds does not change or copy them.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.bounds
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.bds_real)
        self.assertTrue(arrays_share_data(result, self.bds_real),
                        'Bounds do not share data with the assigned array.')

    def test_lazy_bounds(self):
        # Getting lazy bounds realises them.
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        self.assertTrue(coord.has_lazy_bounds())
        result = coord.bounds
        self.assertFalse(coord.has_lazy_bounds())
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, self.bds_real)

    def test_lazy_bounds_with_lazy_points(self):
        # Getting lazy bounds does not fetch the points.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy)
        coord.bounds
        self.assertTrue(coord.has_lazy_points())


class Test_bounds__setter(tests.IrisTest):
    def setUp(self):
        setup_test_arrays(self)

    def test_set_real_bounds(self):
        # Setting new real bounds does not make a copy.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        new_bounds = self.bds_real + 102.3
        coord.bounds = new_bounds
        result = coord.core_bounds()
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, new_bounds)
        self.assertTrue(arrays_share_data(result, new_bounds),
                        'Bounds do not share data with the assigned array.')

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        msg = 'must be compatible with points shape'
        with self.assertRaisesRegexp(ValueError, msg):
            coord.bounds = np.array([1.0, 2.0, 3.0])

    def test_set_lazy_bounds(self):
        # Setting new lazy bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        new_bounds = self.bds_lazy + 102.3
        coord.bounds = new_bounds
        self.assertTrue(coord.has_lazy_bounds())

    def test_set_bounds_with_lazy_points(self):
        # Setting bounds does not change lazy points.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_real)
        new_bounds = self.bds_real + 102.3
        coord.bounds = new_bounds
        self.assertTrue(coord.has_lazy_points())


if __name__ == '__main__':
    tests.main()
