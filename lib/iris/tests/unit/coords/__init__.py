# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.coords` module.

Provides test methods and classes common to
:class:`~iris.tests.unit.coords.test_AuxCoord` and
:class:`~iris.tests.unit.coords.test_DimCoord`.

"""

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import is_lazy_data
from iris.tests import _shared_utils


def _setup_test_arrays(self, shape, masked=False):
    # Create concrete and lazy coordinate points and bounds test arrays,
    # given a desired coord shape.
    # If masked=True, also add masked arrays with some or no masked data,
    # for both points and bounds, lazy and real.
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
    if masked:
        mpoints = ma.array(points)
        self.no_masked_pts_real = mpoints
        self.no_masked_pts_lazy = da.from_array(mpoints, mpoints.shape, asarray=False)
        mpoints = ma.array(mpoints, copy=True)
        mpoints[0] = ma.masked
        self.masked_pts_real = mpoints
        self.masked_pts_lazy = da.from_array(mpoints, mpoints.shape, asarray=False)
        mbounds = ma.array(bounds)
        self.no_masked_bds_real = mbounds
        self.no_masked_bds_lazy = da.from_array(mbounds, mbounds.shape, asarray=False)
        mbounds = ma.array(mbounds, copy=True)
        mbounds[0] = ma.masked
        self.masked_bds_real = mbounds
        self.masked_bds_lazy = da.from_array(mbounds, mbounds.shape, asarray=False)


def is_real_data(array):
    # A parallel to :func:`iris._lazy_data.is_lazy_data`.
    # Not just "not lazy" : ensure it is a 'real' array (i.e. numpy).
    return isinstance(array, np.ndarray)


def is_masked_data(array):
    # Check the array is a masked array.
    return ma.isMaskedArray(array)


def arrays_share_data(a1, a2):
    # Check whether 2 real arrays with the same content view the same data.
    # For an ndarray x, x.base will either be None (if x owns its data) or a
    # reference to the array which owns its data (if x is a view).
    return (
        a1 is a2
        or a1.base is a2
        or a2.base is a1
        or a1.base is a2.base
        and a1.base is not None
    )


def lazyness_string(data):
    # Represent the lazyness of an array as a string.
    return "lazy" if is_lazy_data(data) else "real"


def coords_all_dtypes_and_lazynesses(self, coord_class):
    # Generate coords with all possible types of points and bounds, and all
    # of the given dtypes.
    points_types = ["real", "lazy"]
    bounds_types = ["no", "real", "lazy"]
    # Test a few specific combinations of points+bounds dtypes, including
    # cases where they are different.
    dtype_pairs = [
        (np.float64, np.float64),
        (np.int16, np.int16),
        (np.int16, np.float32),
        (np.float64, np.int32),
    ]
    for pts_dtype, bds_dtype in dtype_pairs:
        for points_type_name in points_types:
            for bounds_type_name in bounds_types:
                pts = np.asarray(self.pts_real, dtype=pts_dtype)
                bds = np.asarray(self.bds_real, dtype=bds_dtype)
                if points_type_name == "lazy":
                    pts = da.from_array(pts, pts.shape)
                if bounds_type_name == "lazy":
                    bds = da.from_array(bds, bds.shape)
                elif bounds_type_name == "no":
                    bds = None
                coord = coord_class(pts, bounds=bds)
                result = (coord, points_type_name, bounds_type_name)
                yield result


class CoordTestMixin:
    def setup_test_arrays(self, shape=(3,), masked=False):
        _setup_test_arrays(self, shape=shape, masked=masked)

    def assert_arrays_share_data(self, a1, a2, msg=None):
        # Check that two arrays are both real, same dtype, and based on the
        # same underlying data (so changing one will change the other).
        self.assert_is_real_array(a1)
        self.assert_is_real_array(a2)
        assert a1.dtype == a2.dtype
        if not msg:
            msg = f"Array {a1} should share data with {a2}"
        assert arrays_share_data(a1, a2), msg

    def assert_arrays_do_not_share_data(self, a1, a2, msg=None):
        if not msg:
            msg = f"Array {a1} should not share data with {a2}"
        assert not arrays_share_data(a1, a2), msg

    def assert_is_real_array(self, array, msg=None):
        # Check that the arg is a real array.
        if not msg:
            msg = f"Array {array} is not a real array"
        assert is_real_data(array), msg

    def assert_is_lazy_array(self, array, msg=None):
        # Check that the arg is a lazy array.
        if not msg:
            msg = f"Array {array} is not a lazy array"
        assert is_lazy_data(array), msg

    def assert_is_masked_array(self, array, msg=None):
        # Check that the arg is a masked array.
        if not msg:
            msg = f"Array {array} is not a masked array"
        assert is_masked_data(array), msg

    def assert_equal_real_arrays_and_dtypes(self, a1, a2):
        # Check that two arrays are real, equal, and have same dtype.
        self.assert_is_real_array(a1)
        self.assert_is_real_array(a2)
        assert a1.dtype == a2.dtype
        _shared_utils.assert_array_equal(a1, a2)

    def assert_equal_lazy_arrays_and_dtypes(self, a1, a2):
        # Check that two arrays are lazy, equal, and have same dtype.
        self.assert_is_lazy_array(a1)
        self.assert_is_lazy_array(a2)
        assert a1.dtype == a2.dtype
        _shared_utils.assert_array_equal(a1.compute(), a2.compute())
