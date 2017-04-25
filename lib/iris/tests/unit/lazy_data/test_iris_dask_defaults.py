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
"""
Test :func:`iris._lazy data._iris_dask_defaults` function.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.context
from iris._lazy_data import _iris_dask_defaults


class Test__iris_dask_defaults(tests.IrisTest):
    def setUp(self):
        self.context = 'dask.context'
        self._globals = 'iris._lazy_data._globals'
        set_options = 'dask.set_options'
        self.patch_set_options = self.patch(set_options)
        get_sync = 'dask.async.get_sync'
        self.patch_get_sync = self.patch(get_sync)

    def test_no_user_options(self):
        test_dict = {}
        with self.patch(self.context, _globals=test_dict):
            _iris_dask_defaults()
            self.assertEqual(dask.context._globals, test_dict)
        self.patch_set_options.assert_called_once_with(get=self.patch_get_sync)

    def test_user_options__pool(self):
        test_dict = {'pool': 5}
        with self.patch(self.context, _globals=test_dict):
            _iris_dask_defaults()
            self.assertEqual(dask.context._globals, test_dict)
        self.assertEqual(self.patch_set_options.call_count, 0)

    def test_user_options__get(self):
        test_dict = {'get': 'threaded'}
        with self.patch(self.context, _globals=test_dict):
            _iris_dask_defaults()
            self.assertEqual(dask.context._globals, test_dict)
        self.assertEqual(self.patch_set_options.call_count, 0)

    def test_user_options__wibble(self):
        # Test a user-specified dask option that does not affect Iris.
        test_dict = {'wibble': 'foo'}
        with self.patch(self.context, _globals=test_dict):
            _iris_dask_defaults()
            self.assertEqual(dask.context._globals, test_dict)
        self.patch_set_options.assert_called_once_with(get=self.patch_get_sync)


if __name__ == '__main__':
    tests.main()
