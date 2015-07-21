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
"""Test function :func:`iris.util.invert_coordinate`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest

import iris
import iris.tests.stock as stock


class TestAll(iris.tests.IrisTest):
    def setUp(self):
        self.cube = stock.geodetic((2, 2))

    def test_inversion_values(self):
        tar = self.cube.data[::-1]

        iris.util.invert_coordinate(self.cube, 'latitude')

        self.assertArrayAlmostEqual(self.cube.data, tar)

        tar = [45., -45.]
        self.assertArrayAlmostEqual(self.cube.coord('latitude').points, tar)

        tar = [[90, 0], [0, -90]]
        self.assertArrayAlmostEqual(self.cube.coord('latitude').bounds, tar)

    def test_aux_coord(self):
        coord = iris.coords.AuxCoord.from_coord(self.cube.coord('latitude'))
        self.cube.remove_coord('latitude')
        self.cube.add_aux_coord(coord, 0)
        msg = ('Only an inversion of a dimension coordinate is supported '
               '\(latitude\)')
        with self.assertRaisesRegexp(RuntimeError, msg):
            iris.util.invert_coordinate(self.cube, 'latitude')

    def test_double_inversion(self):
        # Check that only the intended changes are made by double inversion.
        # We should have exactly the same thing as we started with.
        tar = self.cube.copy()
        iris.util.invert_coordinate(self.cube, 'latitude')
        iris.util.invert_coordinate(self.cube, 'latitude')
        self.assertEqual(self.cube, tar)


if __name__ == '__main__':
    unittest.main()
