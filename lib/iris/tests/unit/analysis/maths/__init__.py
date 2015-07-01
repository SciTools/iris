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
"""Unit tests for the :mod:`iris.analysis.maths` module."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

from abc import ABCMeta, abstractproperty

import numpy as np

from iris.analysis import MEAN
from iris.cube import Cube
import iris.tests.stock as stock


class CubeArithmeticBroadcastingTestMixin(object):
    # A framework for testing the broadcasting behaviour of the various cube
    # arithmetic operations.  (A test for each operation inherits this).
    __metaclass__ = ABCMeta

    @abstractproperty
    def data_op(self):
        # Define an operator to be called, I.E. 'operator.xx'.
        pass

    @abstractproperty
    def cube_func(self):
        # Define an iris arithmetic function to be called
        # I.E. 'iris.analysis.maths.xx'.
        pass

    def test_transposed(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.copy()
        other.transpose()
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        expected_data = self.data_op(cube.data, other.data.T)
        self.assertArrayEqual(res.data, expected_data)

    def test_collapse_zeroth_dim(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed('time', MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # No modification to other.data is needed as numpy broadcasting
        # should be sufficient.
        expected_data = self.data_op(cube.data, other.data)
        # Use assertMaskedArrayEqual as collapsing with MEAN results
        # in a cube with a masked data array.
        self.assertMaskedArrayEqual(res.data, expected_data)

    def test_collapse_all_dims(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed(cube.coords(dim_coords=True), MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # No modification to other.data is needed as numpy broadcasting
        # should be sufficient.
        expected_data = self.data_op(cube.data, other.data)
        # Use assertArrayEqual rather than assertMaskedArrayEqual as
        # collapsing all dims does not result in a masked array.
        self.assertArrayEqual(res.data, expected_data)

    def test_collapse_last_dims(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed(['grid_latitude', 'grid_longitude'], MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # Transpose the dimensions in self.cube that have been collapsed in
        # other to lie at the front, thereby enabling numpy broadcasting to
        # function when applying data operator. Finish by transposing back
        # again to restore order.
        expected_data = self.data_op(cube.data.transpose((2, 3, 0, 1)),
                                     other.data).transpose(2, 3, 0, 1)
        self.assertMaskedArrayEqual(res.data, expected_data)

    def test_collapse_middle_dim(self):
        cube = stock.realistic_4d_no_derived()
        other = cube.collapsed(['model_level_number'], MEAN)
        res = self.cube_func(cube, other)
        self.assertCML(res, checksum=False)
        # Add the collapsed dimension back in via np.newaxis to enable
        # numpy broadcasting to function.
        expected_data = self.data_op(cube.data,
                                     other.data[:, np.newaxis, ...])
        self.assertMaskedArrayEqual(res.data, expected_data)

    def test_slice(self):
        cube = stock.realistic_4d_no_derived()
        for dim in range(cube.ndim):
            keys = [slice(None)] * cube.ndim
            keys[dim] = 3
            other = cube[tuple(keys)]
            res = self.cube_func(cube, other)
            self.assertCML(res, checksum=False)
            # Add the collapsed dimension back in via np.newaxis to enable
            # numpy broadcasting to function.
            keys[dim] = np.newaxis
            expected_data = self.data_op(cube.data,
                                         other.data[tuple(keys)])
            msg = 'Problem broadcasting cubes when sliced on dimension {}.'
            self.assertArrayEqual(res.data, expected_data,
                                  err_msg=msg.format(dim))


class CubeArithmeticMaskingTestMixin(object):
    # A framework for testing the mask handling behaviour of the various cube
    # arithmetic operations.  (A test for each operation inherits this).
    __metaclass__ = ABCMeta

    @abstractproperty
    def data_op(self):
        # Define an operator to be called, I.E. 'operator.xx'.
        pass

    @abstractproperty
    def cube_func(self):
        # Define an iris arithmetic function to be called
        # I.E. 'iris.analysis.maths.xx'.
        pass

    def _test_partial_mask(self, in_place):
        # Helper method for masked data tests.
        dat_a = np.ma.array([2., 2., 2., 2.], mask=[1, 0, 1, 0])
        dat_b = np.ma.array([2., 2., 2., 2.], mask=[1, 1, 0, 0])

        cube_a = Cube(dat_a)
        cube_b = Cube(dat_b)

        com = self.data_op(dat_b, dat_a)
        res = self.cube_func(cube_b, cube_a, in_place=in_place)

        return com, res, cube_b

    def test_partial_mask_in_place(self):
        # Cube in_place arithmetic operation.
        com, res, orig_cube = self._test_partial_mask(True)

        self.assertMaskedArrayEqual(com, res.data, strict=True)
        self.assertIs(res, orig_cube)

    def test_partial_mask_not_in_place(self):
        # Cube arithmetic not an in_place operation.
        com, res, orig_cube = self._test_partial_mask(False)

        self.assertMaskedArrayEqual(com, res.data)
        self.assertIsNot(res, orig_cube)
