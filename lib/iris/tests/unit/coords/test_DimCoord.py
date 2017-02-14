# (C) British Crown Copyright 2013 - 2016, Met Office
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
"""Unit tests for :class:`iris.coords.DimCoord`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.coords import DimCoord
from iris.tests import mock


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.points = [1, 2, 3]

    def _test_ok(self, bounds):
        coord = DimCoord(self.points, bounds=bounds)
        self.assertArrayEqual(coord.points, self.points)
        self.assertArrayEqual(coord.bounds, bounds)

    def test_bounds_contiguous_centred_ok(self):
        bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
        self._test_ok(bounds)

    def test_bounds_contiguous_accumulation_ok(self):
        bounds = [[0, 1], [1, 2], [2, 3]]
        self._test_ok(bounds)

    def test_bounds_overlapping_accumulation_ok(self):
        bounds = [[0, 1], [0, 2], [0, 3]]
        self._test_ok(bounds)

    def test_bounds_overlapping_accumulation_reversed_ok(self):
        bounds = [[1, 0], [2, 0], [3, 0]]
        self._test_ok(bounds)

    def test_bad_bounds(self):
        bounds = [[0, 0.5], [0, 2], [0, 3]]
        with self.assertRaises(ValueError):
            DimCoord(self.points, bounds=bounds)

    def test_bad_bounds_reversed(self):
        bounds = [[0.5, 0], [2, 0], [3, 0]]
        with self.assertRaises(ValueError):
            DimCoord(self.points, bounds=bounds)


class Test_bounds(tests.IrisTest):
    def setUp(self):
        points = [1, 2, 3]
        self.coord = DimCoord(points)

    def test_bounds_contiguous_centred_ok(self):
        bounds = [[0, 1], [1, 2], [2, 3]]
        self.coord.bounds = bounds
        self.assertArrayEqual(self.coord.bounds, bounds)

    def test_bounds_contiguous_accumulation_ok(self):
        bounds = [[0, 1], [1, 2], [2, 3]]
        self.coord.bounds = bounds
        self.assertArrayEqual(self.coord.bounds, bounds)

    def test_bounds_overlapping_accumulation_ok(self):
        bounds = [[0, 1], [0, 2], [0, 3]]
        self.coord.bounds = bounds
        self.assertArrayEqual(self.coord.bounds, bounds)

    def test_bounds_overlapping_accumulation_reversed_ok(self):
        bounds = [[1, 0], [2, 0], [3, 0]]
        self.coord.bounds = bounds
        self.assertArrayEqual(self.coord.bounds, bounds)

    def test_bad_bounds(self):
        bounds = [[0, 0.5], [0, 2], [0, 3]]
        with self.assertRaises(ValueError):
            self.coord.bounds = bounds

    def test_bad_bounds_reversed(self):
        bounds = [[0.5, 0], [2, 0], [3, 0]]
        with self.assertRaises(ValueError):
            self.coord.bounds = bounds


if __name__ == '__main__':
    tests.main()
