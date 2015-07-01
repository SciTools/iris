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
"""Unit tests for the `iris.fileformats.grib.save_messages` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock
from mock import call
import numpy as np

import iris.fileformats.grib as grib


class TestSaveMessages(tests.IrisTest):
    def setUp(self):
        # Create a test object to stand in for a real PPField.
        self.grib_message = gribapi.grib_new_from_samples("GRIB2")

    def test_save(self):
        m = mock.mock_open()
        with mock.patch('__builtin__.open', m, create=True):
            # sending a MagicMock object to gribapi raises an AssertionError
            # as the gribapi code does a type check
            # this is deemed acceptable within the scope of this unit test
            with self.assertRaises(AssertionError):
                grib.save_messages([self.grib_message], 'foo.grib2')
        self.assertTrue(call('foo.grib2', 'wb') in m.mock_calls)

    def test_save_append(self):
        m = mock.mock_open()
        with mock.patch('__builtin__.open', m, create=True):
            # sending a MagicMock object to gribapi raises an AssertionError
            # as the gribapi code does a type check
            # this is deemed acceptable within the scope of this unit test
            with self.assertRaises(AssertionError):
                grib.save_messages([self.grib_message], 'foo.grib2',
                                   append=True)
        self.assertTrue(call('foo.grib2', 'ab') in m.mock_calls)


if __name__ == "__main__":
    tests.main()
