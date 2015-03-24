# (C) British Crown Copyright 2015, Met Office
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
Unit tests for `iris.aux_factory.AuxCoordFactory`.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import biggus
import numpy as np

import iris.coords
from iris.aux_factory import AuxCoordFactory


class Test__nd_points(tests.IrisTest):
    def test_numpy_scalar_cooord(self):
        points = np.arange(1)
        coord = iris.coords.AuxCoord(points)
        result = AuxCoordFactory._nd_points(coord, (), 2)
        expected = points[np.newaxis]
        self.assertArrayEqual(result, expected)

    def test_numpy_simple(self):
        points = np.arange(12).reshape(4, 3)
        coord = iris.coords.AuxCoord(points)
        result = AuxCoordFactory._nd_points(coord, (0, 1), 2)
        expected = points
        self.assertArrayEqual(result, expected)

    def test_numpy_complex(self):
        points = np.arange(12).reshape(4, 3)
        coord = iris.coords.AuxCoord(points)
        result = AuxCoordFactory._nd_points(coord, (3, 2), 5)
        expected = points.T[np.newaxis, np.newaxis, ..., np.newaxis]
        self.assertArrayEqual(result, expected)

    def test_biggus_simple(self):
        raw_points = np.arange(12).reshape(4, 3)
        points = biggus.NumpyArrayAdapter(raw_points)
        coord = iris.coords.AuxCoord(points)
        self.assertIsInstance(coord._points, biggus.Array)
        result = AuxCoordFactory._nd_points(coord, (0, 1), 2)
        # Check we haven't triggered the loading of the coordinate values.
        self.assertIsInstance(coord._points, biggus.Array)
        self.assertIsInstance(result, biggus.Array)
        expected = raw_points
        self.assertArrayEqual(result, expected)

    def test_biggus_complex(self):
        raw_points = np.arange(12).reshape(4, 3)
        points = biggus.NumpyArrayAdapter(raw_points)
        coord = iris.coords.AuxCoord(points)
        self.assertIsInstance(coord._points, biggus.Array)
        result = AuxCoordFactory._nd_points(coord, (3, 2), 5)
        # Check we haven't triggered the loading of the coordinate values.
        self.assertIsInstance(coord._points, biggus.Array)
        self.assertIsInstance(result, biggus.Array)
        expected = raw_points.T[np.newaxis, np.newaxis, ..., np.newaxis]
        self.assertArrayEqual(result, expected)


if __name__ == '__main__':
    tests.main()
