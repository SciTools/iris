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
"""Unit tests for the :func:`iris.analysis.interpolate.linear` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from collections import OrderedDict

import numpy as np

import iris
from iris.analysis.interpolate import linear
from iris.tests import mock
import iris.tests.stock as stock


class Test(tests.IrisTest):
    def setUp(self):
        self.cube = stock.simple_2d()
        self.extrapolation = 'extrapolation_mode'
        self.scheme = mock.Mock(name='linear scheme')

    @mock.patch('iris.analysis.interpolate.Linear', name='linear_patch')
    @mock.patch('iris.cube.Cube.interpolate', name='cube_interp_patch')
    def _assert_expected_call(self, sample_points, sample_points_call,
                              cinterp_patch, linear_patch):
        linear_patch.return_value = self.scheme
        linear(self.cube, sample_points, self.extrapolation)

        linear_patch.assert_called_once_with(self.extrapolation)

        cinterp_patch.assert_called_once_with(sample_points_call, self.scheme)

    def test_sample_point_dict(self):
        # Passing sample_points in the form of a dictionary.
        sample_points_call = [('foo', 0.5), ('bar', 0.5)]
        sample_points = OrderedDict(sample_points_call)
        self._assert_expected_call(sample_points, sample_points_call)

    def test_sample_point_iterable(self):
        # Passing an interable sample_points object.
        sample_points = (('foo', 0.5), ('bar', 0.5))
        sample_points_call = sample_points
        self._assert_expected_call(sample_points, sample_points_call)


class Test_masks(tests.IrisTest):
    def test_mask_retention(self):
        cube = stock.realistic_4d_w_missing_data()
        interp_cube = linear(cube, [('pressure', [850, 950])])
        self.assertIsInstance(interp_cube.data, np.ma.MaskedArray)

        # this value is masked in the input
        self.assertTrue(cube.data.mask[0, 2, 2, 0])
        # and is still masked in the output
        self.assertTrue(interp_cube.data.mask[0, 1, 2, 0])


class TestNDCoords(tests.IrisTest):
    def setUp(self):
        cube = stock.simple_3d_w_multidim_coords()
        cube.add_dim_coord(iris.coords.DimCoord(np.arange(3), 'longitude'), 1)
        cube.add_dim_coord(iris.coords.DimCoord(np.arange(4), 'latitude'), 2)
        cube.data = cube.data.astype(np.float32)
        self.cube = cube

    def test_multi(self):
        # Testing interpolation on specified points on cube with
        # multidimensional coordinates.
        interp_cube = linear(self.cube, {'latitude': 1.5, 'longitude': 1.5})
        self.assertCMLApproxData(interp_cube, ('experimental', 'analysis',
                                               'interpolate',
                                               'linear_nd_2_coords.cml'))

    def test_single_extrapolation(self):
        # Interpolation on the 1d coordinate with extrapolation.
        interp_cube = linear(self.cube, {'wibble': np.float32(1.5)})
        expected = ('experimental', 'analysis', 'interpolate',
                    'linear_nd_with_extrapolation.cml')
        self.assertCMLApproxData(interp_cube, expected)

    def test_single(self):
        # Interpolation on the 1d coordinate.
        interp_cube = linear(self.cube, {'wibble': 20})
        self.assertArrayEqual(np.mean(self.cube.data, axis=0),
                              interp_cube.data)
        self.assertCMLApproxData(interp_cube, ('experimental', 'analysis',
                                               'interpolate', 'linear_nd.cml'))


if __name__ == "__main__":
    tests.main()
