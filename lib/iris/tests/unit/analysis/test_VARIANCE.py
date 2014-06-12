# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the :data:`iris.analysis.VARIANCE` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import biggus
import numpy as np
import numpy.ma as ma

from iris.analysis import VARIANCE
import iris.cube
from iris.coords import DimCoord
import iris.exceptions


class Test_units_func(tests.IrisTest):
    def test(self):
        self.assertIsNotNone(VARIANCE.units_func)
        mul = mock.Mock(return_value=mock.sentinel.new_unit)
        units = mock.Mock(__mul__=mul)
        new_units = VARIANCE.units_func(units)
        # Make sure the VARIANCE units_func tries to square the units.
        mul.assert_called_once_with(units)
        self.assertEqual(new_units, mock.sentinel.new_unit)


class Test_masked(tests.IrisTest):
    def setUp(self):
        self.cube = iris.cube.Cube(ma.masked_equal([1, 2, 3, 4, 5], 3))
        self.cube.add_dim_coord(DimCoord([6, 7, 8, 9, 10], long_name='foo'), 0)

    def test_ma(self):
        # Note: iris.analysis.VARIANCE adds ddof=1
        cube = self.cube.collapsed("foo", VARIANCE)
        self.assertArrayAlmostEqual(cube.data, [3.333333])

    def test_ma_ddof0(self):
        cube = self.cube.collapsed("foo", VARIANCE, ddof=0)
        self.assertArrayEqual(cube.data, [2.5])

    # Pending #1004.
#     def test_biggus(self):
#         self.cube.lazy_data(array=biggus.NumpyArrayAdapter(self.cube.data))
#         cube = self.cube.collapsed("foo", VARIANCE, lazy=True)
#         self.assertArrayAlmostEqual(cube.lazy_data().masked_array(),
#                                     [3.333333])


class Test_lazy_aggregate(tests.IrisTest):
    def test_unsupported_mdtol(self):
        # The VARIANCE aggregator supports lazy_aggregation but does
        # not provide mdtol handling. Check that a TypeError is raised
        # if this unsupported kwarg is specified.
        array = biggus.NumpyArrayAdapter(np.arange(8))
        msg = "unexpected keyword argument 'mdtol'"
        with self.assertRaisesRegexp(TypeError, msg):
            VARIANCE.lazy_aggregate(array, axis=0, mdtol=0.8)

    def test_ddof_one(self):
        array = biggus.NumpyArrayAdapter(np.arange(8))
        var = VARIANCE.lazy_aggregate(array, axis=0, ddof=1)
        self.assertArrayAlmostEqual(var.ndarray(), np.array(6.0))

    def test_ddof_zero(self):
        array = biggus.NumpyArrayAdapter(np.arange(8))
        var = VARIANCE.lazy_aggregate(array, axis=0, ddof=0)
        self.assertArrayAlmostEqual(var.ndarray(), np.array(5.25))


if __name__ == "__main__":
    tests.main()
