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

from iris._lazy_data import _iris_dask_defaults


class Test__iris_dask_defaults(tests.IrisTest):
    def setUp(self):
        dask_opts = 'iris._lazy_data.DASK_OPTS'
        self.mock_opts = self.patch(dask_opts, {})
        self.mock_get_sync = tests.mock.sentinel.get_sync
        get_sync = 'iris._lazy_data.dget_sync'
        self.patch_get_sync = self.patch(get_sync, self.mock_get_sync)
        self.iris_defaults = {'get': self.patch_get_sync}

    def test_startup(self):
        # Dask options for Iris should not be modified on Iris import.
        self.assertDictEqual(self.mock_opts, {})

    def test_no_user_options(self):
        self.patch('dask.context._globals', {})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, self.iris_defaults)

    def test_user_options__pool(self):
        self.patch('dask.context._globals', {'pool': 5})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, {})

    def test_user_options__get(self):
        self.patch('dask.context._globals', {'get': 'threaded'})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, {})

    def test_user_options__wibble(self):
        # Test a user-specified dask option that does not affect Iris.
        self.patch('dask.context._globals', {'wibble': 'foo'})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, self.iris_defaults)

    def test_changed_options__add(self):
        # Check that adding dask options during a session alters Iris dask
        # processing options.
        # Starting condition: no dask options set.
        self.patch('dask.context._globals', {})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, self.iris_defaults)
        # Updated condition: dask option is set.
        self.patch('dask.context._globals', {'get': 'threaded'})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, {})

    def test_changed_options__remove(self):
        # Check that removing dask options during a session alters Iris dask
        # processing options.
        # Starting condition: dask option is set.
        self.patch('dask.context._globals', {'get': 'threaded'})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, {})
        # Updated condition: no dask options set.
        self.patch('dask.context._globals', {})
        _iris_dask_defaults()
        self.assertDictEqual(self.mock_opts, self.iris_defaults)


if __name__ == '__main__':
    tests.main()
