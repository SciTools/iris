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
"""
Unit tests for the :func:`iris.experimental.stratify.relevel` function.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from functools import partial

import numpy as np
from numpy.testing import assert_array_equal

import iris
from iris.coords import AuxCoord, DimCoord
import iris.tests.stock as stock

try:
    import stratify
    from iris.experimental.stratify import relevel
except ImportError:
    stratify = None


@tests.skip_stratify
class Test(tests.IrisTest):
    def setUp(self):
        cube = stock.simple_3d()[:, :1, :1]
        #: The data from which to get the levels.
        self.src_levels = cube.copy()
        #: The data to interpolate.
        self.cube = cube.copy()
        self.cube.rename('foobar')
        self.cube *= 10
        self.coord = self.src_levels.coord('wibble')
        self.axes = (self.coord, self.coord.name(), None, 0)

    def test_broadcast_fail_src_levels(self):
        emsg = 'Cannot broadcast the cube and src_levels'
        data = np.arange(60).reshape(3, 4, 5)
        with self.assertRaisesRegexp(ValueError, emsg):
            relevel(self.cube, AuxCoord(data), [1, 2, 3])

    def test_broadcast_fail_tgt_levels(self):
        emsg = 'Cannot broadcast the cube and tgt_levels'
        data = np.arange(60).reshape(3, 4, 5)
        with self.assertRaisesRegexp(ValueError, emsg):
            relevel(self.cube, self.coord, data)

    def test_standard_input(self):
        for axis in self.axes:
            result = relevel(self.cube,
                             self.src_levels,
                             [-1, 0, 5.5],
                             axis=axis)
            assert_array_equal(result.data.flatten(),
                               np.array([np.nan, 0, 55]))
            expected = DimCoord([-1, 0, 5.5], units=1, long_name='thingness')
            self.assertEqual(expected, result.coord('thingness'))

    def test_non_monotonic(self):
        for axis in self.axes:
            result = relevel(self.cube,
                             self.src_levels,
                             [2, 3, 2],
                             axis=axis)
            assert_array_equal(result.data.flatten(),
                               np.array([20, 30, np.nan]))
            expected = AuxCoord([2, 3, 2], units=1, long_name='thingness')
            self.assertEqual(result.coord('thingness'), expected)

    def test_static_level(self):
        for axis in self.axes:
            result = relevel(self.cube,
                             self.src_levels,
                             [2, 2],
                             axis=axis)
            assert_array_equal(result.data.flatten(), np.array([20, 20]))

    def test_coord_input(self):
        source = AuxCoord(self.src_levels.data)
        source.metadata = self.src_levels.metadata

        for axis in self.axes:
            result = relevel(self.cube,
                             source,
                             [0, 12, 13],
                             axis=axis)
            self.assertEqual(result.shape, (3, 1, 1))
            assert_array_equal(result.data.flatten(), [0, 120, np.nan])

    def test_custom_interpolator(self):
        interpolator = partial(stratify.interpolate, interpolation='nearest')

        for axis in self.axes:
            result = relevel(self.cube,
                             self.src_levels,
                             [-1, 0, 6.5],
                             axis=axis,
                             interpolator=interpolator)
            assert_array_equal(result.data.flatten(),
                               np.array([np.nan, 0, 120]))

    def test_multi_dim_target_levels(self):
        interpolator = partial(stratify.interpolate,
                               interpolation='linear',
                               extrapolation='linear')

        for axis in self.axes:
            result = relevel(self.cube,
                             self.src_levels,
                             self.src_levels.data,
                             axis=axis,
                             interpolator=interpolator)
            assert_array_equal(result.data.flatten(), np.array([0, 120]))
            self.assertCML(result)


if __name__ == "__main__":
    tests.main()
