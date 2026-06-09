# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util._is_circular`."""

import numpy as np
import pytest

from iris.util import _is_circular


@pytest.mark.parametrize(
    "with_bounds",
    [
        pytest.param(True, id="with_bounds"),
        pytest.param(False, id="without_bounds"),
    ],
)
class Test:
    @staticmethod
    def _calc_bounds(points):
        diff = np.diff(points).mean()
        return np.c_[points - 0.5 * diff, points + 0.5 * diff]

    def test_simple(self, with_bounds):
        points = np.arange(12) * 30
        bounds = self._calc_bounds(points) if with_bounds else None
        assert _is_circular(points, 360, bounds)

    def test_negative_diff(self, with_bounds):
        points = (np.arange(96) * -3.749998) + 3.56249908e02
        bounds = self._calc_bounds(points) if with_bounds else None
        assert _is_circular(points, 360, bounds)

    def test_negative_origin(self, with_bounds):
        points = np.arange(-180, 180, 30)
        bounds = self._calc_bounds(points) if with_bounds else None
        assert _is_circular(points, 360, bounds)

    def test_ciruclar_latitude(self, with_bounds):
        points = np.arange(-90, 90, 30)
        bounds = self._calc_bounds(points) if with_bounds else None
        assert _is_circular(points, 180, bounds)  # note the modulus!

    def test_not_circular(self, with_bounds):
        points = np.arange(0, 300, 30)
        bounds = self._calc_bounds(points) if with_bounds else None
        assert not _is_circular(points, 360, bounds)
