# (C) British Crown Copyright 2013 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.abf import ABFField


class MethodCounter(object):
    def __init__(self, method_name):
        self.method_name = method_name
        self.count = 0

    def __enter__(self):
        self.orig_method = getattr(ABFField, self.method_name)

        def new_method(*args, **kwargs):
            self.count += 1
            self.orig_method(*args, **kwargs)

        setattr(ABFField, self.method_name, new_method)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(ABFField, self.method_name, self.orig_method)
        return False


class Test_data(tests.IrisTest):
    def test_single_read(self):
        path = '0000000000000000jan00000'
        field = ABFField(path)

        with mock.patch('iris.fileformats.abf.np.fromfile') as fromfile:
            with MethodCounter('__getattr__') as getattr:
                with MethodCounter('_read') as read:
                    field.data

        fromfile.assert_called_once_with(path, dtype='>u1')
        self.assertEqual(getattr.count, 1)
        self.assertEqual(read.count, 1)


if __name__ == "__main__":
    tests.main()
