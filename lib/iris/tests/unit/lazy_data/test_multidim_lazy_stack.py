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
"""Test function :func:`iris._lazy data.multidim_lazy_stack`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np

from iris._lazy_data import as_concrete_data, as_lazy_data, multidim_lazy_stack


class Test_multidim_lazy_stack(tests.IrisTest):
    def _check(self, stack_shape):
        vals = np.arange(np.prod(stack_shape)).reshape(stack_shape)
        stack = np.empty(stack_shape, 'object')
        # Define the shape of each element in the stack.
        stack_element_shape = (4, 5)
        expected = np.empty(stack_shape + stack_element_shape,
                            dtype=int)
        for index, val in np.ndenumerate(vals):
            stack[index] = as_lazy_data(val * np.ones(stack_element_shape))

            expected[index] = val
        result = multidim_lazy_stack(stack)
        self.assertEqual(result.shape, stack_shape + stack_element_shape)
        self.assertIsInstance(result, da.core.Array)
        result = as_concrete_data(result)
        self.assertArrayAllClose(result, expected)

    def test_0d_lazy_stack(self):
        shape = ()
        self._check(shape)

    def test_1d_lazy_stack(self):
        shape = (2,)
        self._check(shape)

    def test_2d_lazy_stack(self):
        shape = (3, 2)
        self._check(shape)


if __name__ == '__main__':
    tests.main()
