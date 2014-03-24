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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris. If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :mod:`iris.analysis.maths` module."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from abc import ABCMeta, abstractproperty
import numpy as np

import iris
import iris.tests.stock as stock


class _TestBroadcast(tests.IrisTest):
    __metaclass__ = ABCMeta

    @abstractproperty
    def op(self):
        pass

    @abstractproperty
    def func(self):
        pass

    def setUp(self):
        self.cube = stock.simple_4d_with_hybrid_height()
        self.cube._aux_factories = []

    def test_outsermost(self):
        other = self.cube.collapsed(['time'], iris.analysis.MEAN)

        res_cube = self.func(self.cube, other)

        dat = self.op(self.cube.data, other.data)
        com_cube = self.cube.copy(dat)
        com_cube.standard_name = None
        com_cube.units = res_cube.units

        self.assertEqual(res_cube, com_cube)

    def test_innermost(self):
        other = self.cube.collapsed(['grid_latitude', 'grid_longitude'],
                                    iris.analysis.MEAN)
        res_cube = self.func(self.cube, other)

        dat = self.op(self.cube.data.transpose((2, 3, 0, 1)),
                      other.data).transpose(2, 3, 0, 1)
        com_cube = self.cube.copy(dat)
        com_cube.standard_name = None
        com_cube.units = res_cube.units

        self.assertEqual(res_cube, com_cube)

    def test_between(self):
        other = self.cube.collapsed(['model_level_number'], iris.analysis.MEAN)

        res_cube = self.func(self.cube, other)

        dat = self.op(self.cube.data, other.data[:, np.newaxis, ...])
        com_cube = self.cube.copy(dat)
        com_cube.standard_name = None
        com_cube.units = res_cube.units

        self.assertEqual(res_cube, com_cube)
