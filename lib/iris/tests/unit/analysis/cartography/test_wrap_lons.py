# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Unit tests for :func:`iris.analysis.cartography.wrap_lons`."""

import dask.array as da
import numpy as np
import pytest

from iris.analysis.cartography import wrap_lons


class TestWrapLons:
    def test_values(self):
        # The documented behaviour (matches the docstring example).
        result = wrap_lons(np.array([185, 30, -200, 75]), -180, 360)
        np.testing.assert_array_equal(result, [-175.0, 30.0, 160.0, 75.0])

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_floating_dtype_preserved(self, dtype):
        # A floating-point input keeps its dtype rather than being promoted to
        # float64 (see #4119).
        lons = np.array([185, 30, -200, 75], dtype=dtype)
        result = wrap_lons(lons, -180, 360)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, [-175.0, 30.0, 160.0, 75.0])

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_integer_dtype_returns_float64(self, dtype):
        # Integer (and other non-floating) inputs are still returned as float64,
        # because wrapping a discrete range generally yields fractional results.
        lons = np.array([185, 30, -200, 75], dtype=dtype)
        result = wrap_lons(lons, -180, 360)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [-175.0, 30.0, 160.0, 75.0])

    def test_masked_array_preserved(self):
        lons = np.ma.masked_array(
            [185.0, 30.0, -200.0, 75.0], mask=[0, 1, 0, 0], dtype=np.float32
        )
        result = wrap_lons(lons, -180, 360)
        assert isinstance(result, np.ma.MaskedArray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result.mask, [False, True, False, False])

    def test_lazy_input_stays_lazy(self):
        lons = da.from_array(np.array([185, 30, -200, 75], dtype=np.float32))
        result = wrap_lons(lons, -180, 360)
        assert isinstance(result, da.Array)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result.compute(), [-175.0, 30.0, 160.0, 75.0])
