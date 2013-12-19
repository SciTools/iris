# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris.fileformats.abf.ABFField` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from collections import Counter

import mock
import numpy as np

from iris.fileformats.abf import ABFField


class Test_data(tests.IrisTest):
    def test_single_read(self):
        path = '0000000000000000jan00000'
        field = ABFField(path)

        call_counts = Counter()

        orig_getattr = ABFField.__getattr__
        orig_read = ABFField._read

        def new_getattr(self, key):
            call_counts['getattr'] += 1
            orig_getattr(self, key)

        def new_read(self):
            call_counts['read'] += 1
            orig_read(self)

        ABFField.__getattr__ = new_getattr
        ABFField._read = new_read

        with mock.patch('iris.fileformats.abf.np.fromfile') as fromfile:
            field.data

        fromfile.assert_called_once_with(path, dtype='>u1')
        self.assertEqual(call_counts, {'read': 1, 'getattr': 1})


if __name__ == "__main__":
    tests.main()
