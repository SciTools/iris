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
"""Test the function :func:`iris._lazy data.lazy_masked_fill_value`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import lazy_masked_fill_value, _MAX_CHUNK_SIZE


class Test_as_lazy_data(tests.IrisTest):
    def setUp(self):
        shape = (2, 3, 4)
        data = np.arange(24).reshape(shape)
        mask = np.zeros(shape)
        mask[data % 5 == 1] = 1
        self.fill_value = 9999
        self.m = ma.masked_array(data, mask=mask, fill_value=self.fill_value)
        self.dm = da.from_array(self.m, asarray=False,
                                chunks=_MAX_CHUNK_SIZE)

    def test_lazy_masked_ND(self):
        fill_value = lazy_masked_fill_value(self.dm)
        self.assertEqual(fill_value, self.fill_value)

    def test_lazy_masked_0D(self):
        data = self.dm[0, 0, :1]
        fill_value = lazy_masked_fill_value(data)
        self.assertEqual(fill_value, self.fill_value)

    def test_lazy_masked_1D(self):
        data = self.dm[0, 0, :]
        fill_value = lazy_masked_fill_value(data)
        self.assertEqual(fill_value, self.fill_value)

    def test_lazy_masked_2D(self):
        data = self.dm[0, :]
        fill_value = lazy_masked_fill_value(data)
        self.assertEqual(fill_value, self.fill_value)

    def test_real_masked(self):
        fill_value = lazy_masked_fill_value(self.m)
        self.assertEqual(fill_value, self.fill_value)

    def test_lazy_unmasked(self):
        data = da.from_array(self.m.filled(),
                             chunks=_MAX_CHUNK_SIZE)
        fill_value = lazy_masked_fill_value(data)
        self.assertIsNone(fill_value)

    def test_real_unmasked(self):
        data = self.m.filled()
        fill_value = lazy_masked_fill_value(data)
        self.assertIsNone(fill_value)

    def test_data_not_realised(self):
        # Check that only the zero-element slice is realised.
        data = self.dm[0, :]
        lazy_masked_fill_value(data)
        self.assertIsInstance(data, da.core.Array)


if __name__ == '__main__':
    tests.main()
