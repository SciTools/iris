# (C) British Crown Copyright 2018, Met Office
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
"""Unit tests for the :data:`iris.analysis.MIN` aggregator."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma

from iris.analysis import MIN
from iris.cube import Cube
from iris.coords import DimCoord
from iris._lazy_data import as_lazy_data, is_lazy_data


class Test_basics(tests.IrisTest):
    def setUp(self):
        data = np.array([1, 2, 3, 4, 5])
        coord = DimCoord([6, 7, 8, 9, 10], long_name='foo')
        self.cube = Cube(data)
        self.cube.add_dim_coord(coord, 0)
        self.lazy_cube = Cube(as_lazy_data(data))
        self.lazy_cube.add_dim_coord(coord, 0)

    def test_name(self):
        self.assertEqual(MIN.name(), 'minimum')

    def test_collapse(self):
        data = MIN.aggregate(self.cube.data, axis=0)
        self.assertArrayEqual(data, [1])

    def test_lazy(self):
        lazy_data = MIN.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertTrue(is_lazy_data(lazy_data))

    def test_lazy_collapse(self):
        lazy_data = MIN.lazy_aggregate(self.lazy_cube.lazy_data(), axis=0)
        self.assertArrayEqual(lazy_data.compute(), [1])


class Test_masked(tests.IrisTest):
    def setUp(self):
        self.cube = Cube(ma.masked_less([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name='foo'), 0)

    def test_ma(self):
        data = MIN.aggregate(self.cube.data, axis=0)
        self.assertArrayEqual(data, [3])


class Test_lazy_masked(tests.IrisTest):
    def setUp(self):
        masked_data = ma.masked_less([1, 2, 3, 4, 5], 3)
        self.cube = Cube(as_lazy_data(masked_data))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name='foo'), 0)

    def test_lazy_ma(self):
        lazy_data = MIN.lazy_aggregate(self.cube.lazy_data(), axis=0)
        self.assertTrue(is_lazy_data(lazy_data))
        self.assertArrayEqual(lazy_data.compute(), [3])


class Test_aggregate_shape(tests.IrisTest):
    def test(self):
        shape = ()
        kwargs = dict()
        self.assertTupleEqual(MIN.aggregate_shape(**kwargs), shape)
        kwargs = dict(wibble='wobble')
        self.assertTupleEqual(MIN.aggregate_shape(**kwargs), shape)


if __name__ == "__main__":
    tests.main()
