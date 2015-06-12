# (C) British Crown Copyright 2010 - 2015, Met Office
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
"""Test function :func:`iris.util.describe_diff`."""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import io

import numpy as np

import iris.cube
from iris.util import describe_diff


class Test(iris.tests.IrisTest):
    def setUp(self):
        self.cube_a = iris.cube.Cube([])
        self.cube_b = self.cube_a.copy()

    def _compare_result(self, cube_a, cube_b):
        result_bio = io.BytesIO()
        describe_diff(cube_a, cube_b, output_file=result_bio)
        return result_bio.getvalue()

    def test_noncommon_array_attributes(self):
        # test non-common array attribute
        self.cube_a.attributes['test_array'] = np.array([1, 2, 3])
        return_str = self._compare_result(self.cube_a, self.cube_b)
        self.assertString(return_str, ['compatible_cubes.str.txt'])

    def test_same_array_attributes(self):
        # test matching array attribute
        self.cube_a.attributes['test_array'] = np.array([1, 2, 3])
        self.cube_b.attributes['test_array'] = np.array([1, 2, 3])
        return_str = self._compare_result(self.cube_a, self.cube_b)
        self.assertString(return_str, ['compatible_cubes.str.txt'])

    def test_different_array_attributes(self):
        # test non-matching array attribute
        self.cube_a.attributes['test_array'] = np.array([1, 2, 3])
        self.cube_b.attributes['test_array'] = np.array([1, 7, 3])
        return_str = self._compare_result(self.cube_a, self.cube_b)
        self.assertString(
            return_str,
            ['unit', 'util', 'describe_diff',
             'incompatible_array_attrs.str.txt'])


if __name__ == '__main__':
    tests.main()
