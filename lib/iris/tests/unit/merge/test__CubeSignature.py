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
"""Unit tests for the `iris._merge._CubeSignature` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

import iris.exceptions
from iris._merge import _CubeSignature as CubeSig


class Test_match__fill_value(tests.IrisTest):
    def setUp(self):
        self.defn = mock.Mock(standard_name=mock.sentinel.standard_name,
                              long_name=mock.sentinel.long_name,
                              var_name=mock.sentinel.var_name,
                              units=mock.sentinel.units,
                              attributes=mock.sentinel.attributes,
                              cell_methods=mock.sentinel.cell_methods)
        self.data_shape = mock.sentinel.data_shape
        self.data_type = mock.sentinel.data_type

    def test_non_nan_fill_value_equal(self):
        sig1 = CubeSig(self.defn, self.data_shape, self.data_type, 10)
        sig2 = CubeSig(self.defn, self.data_shape, self.data_type, 10)
        self.assertTrue(sig1.match(sig2, True))
        self.assertTrue(sig1.match(sig2, False))
        self.assertTrue(sig2.match(sig1, True))
        self.assertTrue(sig2.match(sig1, False))

    def test_non_nan_fill_value_unequal(self):
        sig1 = CubeSig(self.defn, self.data_shape, self.data_type, 10)
        sig2 = CubeSig(self.defn, self.data_shape, self.data_type, 20)
        with self.assertRaises(iris.exceptions.MergeError):
            sig1.match(sig2, True)
        self.assertFalse(sig1.match(sig2, False))
        with self.assertRaises(iris.exceptions.MergeError):
            sig2.match(sig1, True)
        self.assertFalse(sig2.match(sig1, False))

    def test_nan_fill_value_equal(self):
        sig1 = CubeSig(self.defn, self.data_shape, self.data_type, np.nan)
        sig2 = CubeSig(self.defn, self.data_shape, self.data_type, np.nan)
        self.assertTrue(sig1.match(sig2, True))
        self.assertTrue(sig1.match(sig2, False))
        self.assertTrue(sig2.match(sig1, True))
        self.assertTrue(sig2.match(sig1, False))

    def test_nan_fill_value_unequal(self):
        sig1 = CubeSig(self.defn, self.data_shape, self.data_type, np.nan)
        sig2 = CubeSig(self.defn, self.data_shape, self.data_type, 10)
        with self.assertRaises(iris.exceptions.MergeError):
            sig1.match(sig2, True)
        self.assertFalse(sig1.match(sig2, False))
        with self.assertRaises(iris.exceptions.MergeError):
            sig2.match(sig1, True)
        self.assertFalse(sig2.match(sig1, False))


if __name__ == '__main__':
    tests.main()
