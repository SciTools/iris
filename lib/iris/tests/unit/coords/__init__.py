# (C) British Crown Copyright 2013 - 2017, Met Office
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
Unit tests for the :mod:`iris.coords` module.

Provides test methods and classes common to test_AuxCoord and test_DimCoord.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import dask.array as da
import numpy as np

from iris._lazy_data import is_lazy_data


def setup_test_arrays(self, shape):
    # Create standard coordinate points and bounds test arrays,
    # given a desired coord shape.
    # Also create lazy versions, and save all on the 'self' object.
    n_pts = np.prod(shape)
    # Note: the values must be integral for testing integer dtypes.
    points = 10.0 * np.arange(n_pts, dtype=float).reshape(shape)
    lower = points - 2.0
    upper = points + 2.0
    bounds = np.stack((lower, upper), axis=-1)
    self.pts_real = points
    self.pts_lazy = da.from_array(points, points.shape)
    self.bds_real = bounds
    self.bds_lazy = da.from_array(bounds, bounds.shape)


def is_real_data(array):
    # A parallel to 'is_lazy_data'.
    # Not just "not lazy" : ensure it is a 'real' array (i.e. numpy).
    return isinstance(array, np.ndarray)


def arrays_share_data(a1, a2):
    # Check whether 2 real arrays with the same content view the same data.
    # Notes:
    # *  !! destructive !!
    # *  requires that array contents are initially identical
    # *  forces a1 to be writeable and modifies it
    assert np.all(a1 == a2)
    a1.flags.writeable = True
    a1 += np.array(1.0, dtype=a1.dtype)
    return np.all(a1 == a2)


def lazyness_string(data):
    # Represent the lazyness of an array as a string.
    return 'lazy' if is_lazy_data(data) else 'real'


def coords_all_dtypes_and_lazynesses(self, coord_class,
                                     dtypes=(np.float64, np.int16)):
    # Generate coords with all possible types of points and bounds, and all
    # of the given dtypes.
    points_types = ['real', 'lazy']
    bounds_types = ['no', 'real', 'lazy']
    for dtype in dtypes:
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
                coord = coord_class(pts, bounds=bds)
                result = (coord, points_type_name, bounds_type_name)
                yield result


class CoordTestMixin(object):
    def setupTestArrays(self, shape=(3,)):
        setup_test_arrays(self, shape=shape)

    def assertArraysShareData(self, a1, a2, *args, **kwargs):
        # Check that two arrays are both real, same dtype, and based on the
        # same underlying data (so changing one will change the other).
        self.assertIsRealArray(a1)
        self.assertIsRealArray(a2)
        self.assertEqual(a1.dtype, a2.dtype)
        self.assertTrue(arrays_share_data(a1, a2), *args, **kwargs)

    def assertArraysDoNotShareData(self, a1, a2, *args, **kwargs):
        self.assertFalse(arrays_share_data(a1, a2), *args, **kwargs)

    def assertIsRealArray(self, array, *args, **kwargs):
        # Check that the arg is a real array.
        self.assertTrue(is_real_data(array), *args, **kwargs)

    def assertIsLazyArray(self, array, *args, **kwargs):
        # Check that the arg is a lazy array.
        self.assertTrue(is_lazy_data(array), *args, **kwargs)

    def assertEqualRealArraysAndDtypes(self, a1, a2, *args, **kwargs):
        # Check that two arrays are real, equal, and have same dtype.
        self.assertIsRealArray(a1)
        self.assertIsRealArray(a2)
        self.assertEqual(a1.dtype, a2.dtype)
        self.assertArrayEqual(a1, a2)

    def assertEqualLazyArraysAndDtypes(self, a1, a2, *args, **kwargs):
        # Check that two arrays are lazy, equal, and have same dtype.
        self.assertIsLazyArray(a1)
        self.assertIsLazyArray(a2)
        self.assertEqual(a1.dtype, a2.dtype)
        self.assertArrayEqual(a1.compute(), a2.compute())
