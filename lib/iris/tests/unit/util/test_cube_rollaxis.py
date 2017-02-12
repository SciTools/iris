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
"""Test function :func:`iris.util._cube_rollaxis`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import unittest

import iris
from iris.util import _cube_rollaxis


class Test(tests.IrisTest):

    def setUp(self):

        arr = np.ones((3, 4, 5, 6))
        self.cube = iris.cube.Cube(arr)

    def test_numpy_rollaxis_docstring_examples(self):
        '''
        These examples are all taken from the numpy.rollaxis
        docstring, to check this function behaves in a
        similar way.
        '''

        cube = self.cube.copy()
        _cube_rollaxis(cube, 0, start=0)
        self.assertEqual(cube.shape, self.cube.shape)

        cube = self.cube.copy()
        _cube_rollaxis(cube, 3, start=1)
        self.assertEqual(cube.shape, (3, 6, 4, 5))

        cube = self.cube.copy()
        _cube_rollaxis(cube, 2)
        self.assertEqual(cube.shape, (5, 3, 4, 6))

        cube = self.cube.copy()
        _cube_rollaxis(cube, 1, 4)
        self.assertEqual(cube.shape, (3, 5, 6, 4))

    def test_with_valid_negative_numbers_is_same_as_numpy_rollaxis(self):

        cube = self.cube.copy()
        _cube_rollaxis(cube, -1, 2)
        self.assertEqual(cube.shape, (3, 4, 6, 5))

        cube = self.cube.copy()
        _cube_rollaxis(cube, 0, -1)
        self.assertEqual(cube.shape, (4, 5, 3, 6))

    def test_incorrect_argument_values_is_same_as_numpy_rollaxis(self):

        self.assertRaises(ValueError, _cube_rollaxis, self.cube, -5)
        self.assertRaises(ValueError, _cube_rollaxis, self.cube, 5)
        self.assertRaises(ValueError, _cube_rollaxis, self.cube, 1, -5)
        self.assertRaises(ValueError, _cube_rollaxis, self.cube, 1, 5)


if __name__ == '__main__':
    unittest.main()
