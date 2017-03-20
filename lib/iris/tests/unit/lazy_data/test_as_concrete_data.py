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
"""Test function :func:`iris._lazy data.as_concrete_data`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import numpy.ma as ma
import dask.array as da

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
from iris.tests import mock


class Test_as_concrete_data(tests.IrisTest):
    def test_concrete_input_data(self):
        data = np.arange(24).reshape((4, 6))
        result = as_concrete_data(data)
        self.assertIs(data, result)
        self.assertFalse(is_lazy_data(result))

    def test_concrete_masked_input_data(self):
        data = ma.masked_array([10, 12, 8, 2], mask=[True, True, False, True])
        result = as_concrete_data(data)
        self.assertIs(data, result)
        self.assertFalse(is_lazy_data(result))

    def test_lazy_data(self):
        # Minimal testing as as_concrete_data is a wrapper to
        # convert_nans_array
        data = np.arange(24).reshape((2, 12))
        lazy_array = as_lazy_data(data)
        sentinel = mock.sentinel.data
        with mock.patch('iris._lazy_data.convert_nans_array') as conv_nans:
            conv_nans.return_value = sentinel
            result = as_concrete_data(lazy_array)
        self.assertEqual(sentinel, result)

        # Check call to convert_nans_array
        conv_nans.assert_called_once()
        args, kwargs = conv_nans.call_args
        arg, = args
        self.assertFalse(is_lazy_data(arg))
        self.assertArrayEqual(arg, data)
        self.assertEqual(kwargs, {'result_dtype': None,
                                  'nans_replacement': None})

    def test_lazy_data_pass_thru_kwargs(self):
        # Minimal testing as as_concrete_data is a wrapper to
        # convert_nans_array
        data = np.arange(24).reshape((2, 12))
        lazy_array = as_lazy_data(data)
        sentinel = mock.sentinel.data
        with mock.patch('iris._lazy_data.convert_nans_array') as conv_nans:
            conv_nans.return_value = sentinel
            result = as_concrete_data(lazy_array, nans_replacement=7,
                                      result_dtype=np.int16)
        self.assertEqual(sentinel, result)

        # Check call to convert_nans_array
        conv_nans.assert_called_once()
        args, kwargs = conv_nans.call_args
        arg, = args
        self.assertFalse(is_lazy_data(arg))
        self.assertArrayEqual(arg, data)
        self.assertEqual(kwargs, {'nans_replacement': 7,
                                  'result_dtype': np.int16, })


if __name__ == '__main__':
    tests.main()
