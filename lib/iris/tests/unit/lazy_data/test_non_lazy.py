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
"""Test function :func:`iris._lazy data.non_lazy`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris._lazy_data import as_lazy_data, is_lazy_data, non_lazy


class Test_non_lazy(tests.IrisTest):
    def setUp(self):
        self.array = np.arange(8).reshape(2, 4)
        self.lazy_array = as_lazy_data(self.array)
        self.func = non_lazy(lambda array: array.sum(axis=0))
        self.func_result = [4, 6, 8, 10]

    def test_lazy_input(self):
        result = self.func(self.lazy_array)
        self.assertFalse(is_lazy_data(result))
        self.assertArrayEqual(result, self.func_result)

    def test_non_lazy_input(self):
        # Check that a non-lazy input doesn't trip up the functionality.
        result = self.func(self.array)
        self.assertFalse(is_lazy_data(result))
        self.assertArrayEqual(result, self.func_result)


if __name__ == '__main__':
    tests.main()
