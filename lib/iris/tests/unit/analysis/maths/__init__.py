# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for the :mod:`iris.analysis.maths` module."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from abc import ABCMeta, abstractproperty
import numpy as np

import iris


class _TestValue(tests.IrisTest):
    __metaclass__ = ABCMeta

    @abstractproperty
    def op(self):
        pass

    @abstractproperty
    def func(self):
        pass

    def test_div_partial_mask(self):
        # Testing which
        dat_a = np.ma.array([2., 2., 2., 2.], mask=[1, 0, 1, 0])
        dat_b = np.ma.array([2., 2., 2., 2.], mask=[1, 1, 0, 0])

        cube_a = iris.cube.Cube(dat_a)
        cube_b = iris.cube.Cube(dat_b)

        com = self.op(dat_b, dat_a)
        res = self.func(cube_b, cube_a).data

        self.assertMaskedArrayEqual(com, res)

    def test_div_zero_unmasked(self):
        # Ensure cube behaviour matches numpy
        dat_a = np.array([0., 0., 0., 0.])
        dat_b = np.array([2., 2., 2., 2.])

        cube_a = iris.cube.Cube(dat_a)
        cube_b = iris.cube.Cube(dat_b)

        com = self.op(dat_b, dat_a)
        res = self.func(cube_b, cube_a).data

        self.assertArrayEqual(com, res)

    def test_div_zero_masked(self):
        dat_a = np.ma.array([0., 0., 0., 0.], mask=False)
        dat_b = np.ma.array([2., 2., 2., 2.], mask=False)

        cube_a = iris.cube.Cube(dat_a)
        cube_b = iris.cube.Cube(dat_b)

        com = self.op(dat_b, dat_a)
        res = self.func(cube_b, cube_a).data

        self.assertMaskedArrayEqual(com, res)


class _TestInplace(tests.IrisTest):
    __metaclass__ = ABCMeta

    @abstractproperty
    def func(self):
        pass

    def setUp(self):
        dat_a = np.ma.array([1., 1., 1., 1.], mask=False)
        dat_b = np.ma.array([2., 2., 2., 2.], mask=False)

        self.cube_a = iris.cube.Cube(dat_a)
        self.cube_b = iris.cube.Cube(dat_b)

    def test_inplace_false(self):
        res = self.func(self.cube_b, self.cube_a, in_place=False)
        self.assertIsNot(res, self.cube_b)

    def test_inplace_default(self):
        res = self.func(self.cube_b, self.cube_a)
        self.assertIsNot(res, self.cube_b)

    def test_inplace_true(self):
        res = self.func(self.cube_b, self.cube_a, in_place=True)
        self.assertIs(res, self.cube_b)
