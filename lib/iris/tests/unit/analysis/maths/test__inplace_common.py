# (C) British Crown Copyright 2017, Met Office
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
"""Unit tests for the :func:`iris.analysis.maths._inplace_common` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.cube import Cube
import numpy as np

from iris.analysis.maths import _inplace_common


class Test(tests.IrisTest):
    def setUp(self):
        self.scalar_int = 5
        self.scalar_float = 5.5

        self.float_data = np.array([8, 9], dtype=np.float64)
        self.int_data = np.array([9, 8], dtype=np.int64)

        self.float_cube = Cube(self.float_data)
        self.int_cube = Cube(self.int_data)

        self.op = 'addition'
        self.emsg = 'Cannot perform inplace {}'.format(self.op)

    def test_float_cubes(self):
        result = _inplace_common(self.float_cube, self.float_cube, self.op)
        self.assertIsNone(result)

    def test_int_cubes(self):
        result = _inplace_common(self.int_cube, self.int_cube, self.op)
        self.assertIsNone(result)

    def test_float_cube_int_cube(self):
        result = _inplace_common(self.float_cube, self.int_cube, self.op)
        self.assertIsNone(result)

    def test_int_cube_float_cube(self):
        with self.assertRaisesRegexp(ArithmeticError, self.emsg):
            _inplace_common(self.int_cube, self.float_cube, self.op)

    def test_float_cube__scalar_int(self):
        result = _inplace_common(self.float_cube, self.scalar_int, self.op)
        self.assertIsNone(result)

    def test_float_cube__scalar_float(self):
        result = _inplace_common(self.float_cube, self.scalar_float, self.op)
        self.assertIsNone(result)

    def test_float_cube__int_array(self):
        result = _inplace_common(self.float_cube, self.int_data, self.op)
        self.assertIsNone(result)

    def test_float_cube__float_array(self):
        result = _inplace_common(self.float_cube, self.float_data, self.op)
        self.assertIsNone(result)

    def test_int_cube__scalar_int(self):
        result = _inplace_common(self.int_cube, self.scalar_int, self.op)
        self.assertIsNone(result)

    def test_int_cube__scalar_float(self):
        with self.assertRaisesRegexp(ArithmeticError, self.emsg):
            _inplace_common(self.int_cube, self.scalar_float, self.op)

    def test_int_cube__int_array(self):
        result = _inplace_common(self.int_cube, self.int_cube, self.op)
        self.assertIsNone(result)

    def test_int_cube__float_array(self):
        with self.assertRaisesRegexp(ArithmeticError, self.emsg):
            _inplace_common(self.int_cube, self.float_data, self.op)


if __name__ == "__main__":
    tests.main()
