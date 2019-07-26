# (C) British Crown Copyright 2017 - 2019, Met Office
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
"""Test function :func:`iris._lazy data.is_lazy_data`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np

from iris._lazy_data import is_lazy_data


class Test_is_lazy_data(tests.IrisTest):
    def test_lazy(self):
        values = np.arange(30).reshape((2, 5, 3))
        lazy_array = da.from_array(values, chunks='auto')
        self.assertTrue(is_lazy_data(lazy_array))

    def test_real(self):
        real_array = np.arange(24).reshape((2, 3, 4))
        self.assertFalse(is_lazy_data(real_array))


if __name__ == '__main__':
    tests.main()
