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
"""Unit tests for the :func:`iris.analysis.maths.exponentiate` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
from iris.tests.stock import simple_2d

import iris
from iris.analysis.maths import exponentiate
import numpy as np


class Test_exponentiate(tests.IrisTest):
    def setUp(self):
        self.cube = simple_2d(with_bounds=False)
        self.exponent = 2
        self.expected = self.cube.data ** self.exponent

    def test_basic(self):
        result = exponentiate(self.cube, self.exponent)
        self.assertArrayEqual(result.data, self.expected)

    def test_masked(self):
        cube = self.cube.copy()
        mask = cube.data % 3 == 0
        masked_data = np.ma.masked_array(cube.data, mask)
        cube.data = masked_data
        expected = masked_data ** 2
        result = exponentiate(cube, self.exponent)
        self.assertMaskedArrayEqual(result.data, expected)

    @tests.skip_data
    def test_lazy_data(self):
        # Confirm that the cube's lazy data is preserved through the operation.
        test_data = tests.get_data_path(('PP', 'simple_pp', 'global.pp'))
        cube = iris.load_cube(test_data)
        exponentiate(cube, self.exponent, in_place=True)
        self.assertTrue(cube.has_lazy_data())


if __name__ == "__main__":
    tests.main()
