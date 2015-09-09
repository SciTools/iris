# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Unit tests for the :func:`iris.analysis.maths.divide` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import operator

from iris.analysis.maths import divide
from iris.cube import Cube
from iris.tests.unit.analysis.maths import \
    CubeArithmeticBroadcastingTestMixin, CubeArithmeticMaskingTestMixin


class TestBroadcasting(tests.IrisTest, CubeArithmeticBroadcastingTestMixin):
    @property
    def data_op(self):
        try:
            return operator.div
        except AttributeError:
            return operator.truediv

    @property
    def cube_func(self):
        return divide


class TestMasking(tests.IrisTest, CubeArithmeticMaskingTestMixin):
    @property
    def data_op(self):
        try:
            return operator.div
        except AttributeError:
            return operator.truediv

    @property
    def cube_func(self):
        return divide

    def test_unmasked_div_zero(self):
        # Ensure cube behaviour matches numpy operator behaviour for the
        # handling of arrays containing 0.
        dat_a = np.array([0., 0., 0., 0.])
        dat_b = np.array([2., 2., 2., 2.])

        cube_a = Cube(dat_a)
        cube_b = Cube(dat_b)

        com = self.data_op(dat_b, dat_a)
        res = self.cube_func(cube_b, cube_a).data

        self.assertArrayEqual(com, res)

    def test_masked_div_zero(self):
        # Ensure cube behaviour matches numpy operator behaviour for the
        # handling of arrays containing 0.
        dat_a = np.ma.array([0., 0., 0., 0.], mask=False)
        dat_b = np.ma.array([2., 2., 2., 2.], mask=False)

        cube_a = Cube(dat_a)
        cube_b = Cube(dat_b)

        com = self.data_op(dat_b, dat_a)
        res = self.cube_func(cube_b, cube_a).data

        self.assertMaskedArrayEqual(com, res, strict=True)


if __name__ == "__main__":
    tests.main()
