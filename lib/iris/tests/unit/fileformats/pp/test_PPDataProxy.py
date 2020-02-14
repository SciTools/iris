# (C) British Crown Copyright 2014 - 2019, Met Office
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
"""Unit tests for the `iris.fileformats.pp.PPDataProxy` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.fileformats.pp import PPDataProxy, SplittableInt
from iris.tests import mock


class Test_lbpack(tests.IrisTest):
    def test_lbpack_SplittableInt(self):
        lbpack = mock.Mock(spec_set=SplittableInt)
        proxy = PPDataProxy(None, None, None, None,
                            None, lbpack, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIs(proxy.lbpack, lbpack)

    def test_lbpack_raw(self):
        lbpack = 4321
        proxy = PPDataProxy(None, None, None, None,
                            None, lbpack, None, None)
        self.assertEqual(proxy.lbpack, lbpack)
        self.assertIsNot(proxy.lbpack, lbpack)
        self.assertIsInstance(proxy.lbpack, SplittableInt)
        self.assertEqual(proxy.lbpack.n1, lbpack % 10)
        self.assertEqual(proxy.lbpack.n2, lbpack // 10 % 10)
        self.assertEqual(proxy.lbpack.n3, lbpack // 100 % 10)
        self.assertEqual(proxy.lbpack.n4, lbpack // 1000 % 10)

    def test__getitem__emptyslice(self):
        # Check that indexing with an "empty" slice will *not* open and read the file.
        # This is necessary because, since Dask 2.0, the "from_array" function takes
        # a zero-length slice of its array argument, to capture array metadata.
        test_shape = (3, 4, 2, 5, 6, 3, 7)
        test_dtype = np.dtype(np.float32)
        proxy = PPDataProxy(shape=test_shape, src_dtype=test_dtype,
                            path=None, offset=None, data_len=None, lbpack=None,
                            boundary_packing=None, mdi=None)

        builtin_open_func_name = '{}.open'.format(__name__)
        with self.patch(builtin_open_func_name) as mock_fileopen:
            # Test indexing with "empty" slices, "normal" slices and integers.
            result = proxy[1:3, 2, 0:0, :, 1:1, :100]

        self.assertEqual(mock_fileopen.called, False)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, test_dtype)
        self.assertEqual(result.shape, (2, 0, 5, 0, 3, 7))


if __name__ == '__main__':
    tests.main()
