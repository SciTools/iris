# (C) British Crown Copyright 2015, Met Office
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
"""Test function :func:`iris.util.column_slices_generator`."""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.util import column_slices_generator


class Test_int_types(tests.IrisTest):
    def _test(self, key):
        full_slice = (key,)
        ndims = 1
        mapping, iterable = column_slices_generator(full_slice, ndims)
        self.assertEqual(mapping, {0: None, None: None})
        self.assertEqual(list(iterable), [(0,)])

    def test_int(self):
        self._test(0)

    def test_int_32(self):
        self._test(np.int32(0))

    def test_int_64(self):
        self._test(np.int64(0))


if __name__ == '__main__':
    tests.main()
