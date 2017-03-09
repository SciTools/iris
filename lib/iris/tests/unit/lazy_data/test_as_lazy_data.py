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
"""Test the function :func:`iris._lazy data.as_lazy_data`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import dask.array as da

from iris._lazy_data import as_lazy_data, _MAX_CHUNK_SIZE


class Test_as_lazy_data(tests.IrisTest):
    def test_lazy(self):
        data = da.from_array(np.arange(24).reshape((2, 3, 4)),
                             chunks=_MAX_CHUNK_SIZE)
        result = as_lazy_data(data)
        self.assertIsInstance(result, da.core.Array)

    def test_real(self):
        data = np.arange(24).reshape((2, 3, 4))
        result = as_lazy_data(data)
        self.assertIsInstance(result, da.core.Array)

    def test_masked(self):
        data = np.ma.masked_greater(np.arange(24), 10)
        result = as_lazy_data(data)
        self.assertIsInstance(result, da.core.Array)

    def test_non_default_chunks(self):
        data = np.arange(24)
        chunks = 12
        lazy_data = as_lazy_data(data, chunks=chunks)
        result, = np.unique(lazy_data.chunks)
        self.assertEqual(result, chunks)

    def test_non_default_chunks__chunks_already_set(self):
        chunks = 12
        data = da.from_array(np.arange(24), chunks=chunks)
        lazy_data = as_lazy_data(data)
        result, = np.unique(lazy_data.chunks)
        self.assertEqual(result, chunks)


if __name__ == '__main__':
    tests.main()
