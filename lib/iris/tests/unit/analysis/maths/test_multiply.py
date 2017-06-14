# (C) British Crown Copyright 2014 - 2017, Met Office
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
"""Unit tests for the :func:`iris.analysis.maths.multiply` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import operator

from iris.analysis.maths import multiply
from iris.tests.unit.analysis.maths import \
    CubeArithmeticBroadcastingTestMixin, CubeArithmeticCoordsTest, \
    CubeArithmeticMaskedConstantTestMixin, CubeArithmeticMaskingTestMixin


@tests.skip_data
@tests.iristest_timing_decorator
class TestBroadcasting(tests.IrisTest_nometa,
                       CubeArithmeticBroadcastingTestMixin):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


@tests.iristest_timing_decorator
class TestMasking(tests.IrisTest_nometa, CubeArithmeticMaskingTestMixin):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


class TestCoordMatch(CubeArithmeticCoordsTest):
    def test_no_match(self):
        cube1, cube2 = self.SetUpNonMatching()
        with self.assertRaises(ValueError):
            multiply(cube1, cube2)

    def test_reversed_points(self):
        cube1, cube2 = self.SetUpReversed()
        with self.assertRaises(ValueError):
            multiply(cube1, cube2)


@tests.iristest_timing_decorator
class TestMaskedConstant(tests.IrisTest_nometa,
                         CubeArithmeticMaskedConstantTestMixin):
    @property
    def data_op(self):
        return operator.mul

    @property
    def cube_func(self):
        return multiply


if __name__ == "__main__":
    tests.main()
