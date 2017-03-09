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
"""Test function :func:`iris._lazy data.multidim_daskstack`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import dask.array as da

from iris._lazy_data import as_lazy_data, multidim_daskstack


class Test_multidim_daskstack(tests.IrisTest):
    def test_0d_dask_stack(self):
        value = 4
        data = np.array(as_lazy_data(np.array(value)), dtype=object)
        result = multidim_daskstack(data)
        self.assertEqual(result, value)

    def test_1d_dask_stack(self):
        vals = [4, 11]
        data = np.array([as_lazy_data(val*np.ones((3, 3))) for val in vals],
                        dtype=object)
        result = multidim_daskstack(data)
        self.assertEqual(result.shape, (2, 3, 3))
        self.assertArrayEqual(result[:, 0, 0], np.array(vals))

    def test_2d_dask_stack(self):
        vals = [4, 8, 11]
        data = np.array([as_lazy_data(val*np.ones((2, 2))) for val in vals],
                        dtype=object)
        data = data.reshape(3, 1, 2, 2)
        result = multidim_daskstack(data)
        self.assertEqual(result.shape, (3, 1, 2, 2))
        self.assertArrayEqual(result[:, :, 0, 0], np.array(vals).reshape(3, 1))


if __name__ == '__main__':
    tests.main()
