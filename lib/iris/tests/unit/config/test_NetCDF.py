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
"""Unit tests for the `iris.config.NetCDF` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import warnings

import iris.config


class Test(tests.IrisTest):
    def setUp(self):
        self.options = iris.config.NetCDF()

    def test_basic(self):
        self.assertFalse(self.options.conventions_override)

    def test_enabled(self):
        self.options.conventions_override = True
        self.assertTrue(self.options.conventions_override)

    def test_bad_value(self):
        # A bad value should be ignored and replaced with the default value.
        bad_value = 'wibble'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.options.conventions_override = bad_value
        self.assertFalse(self.options.conventions_override)
        exp_wmsg = 'Attempting to set invalid value {!r}'.format(bad_value)
        six.assertRegex(self, str(w[0].message), exp_wmsg)

    def test__contextmgr(self):
        with self.options.context(conventions_override=True):
            self.assertTrue(self.options.conventions_override)
        self.assertFalse(self.options.conventions_override)


if __name__ == '__main__':
    tests.main()
