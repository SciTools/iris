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
"""Unit tests for the `iris.fileformats.pp.save_fields` function."""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
from mock import call
import numpy as np

import iris.fileformats.pp as pp


def asave(afilehandle):
    afilehandle.write('saved')


class TestSaveFields(tests.IrisTest):
    def setUp(self):
        # Create a test object to stand in for a real PPField.
        self.pp_field = mock.MagicMock(spec=pp.PPField3)
        # Add minimal content required by the pp.save operation.
        self.pp_field.HEADER_DEFN = pp.PPField3.HEADER_DEFN
        self.pp_field.data = np.zeros((1, 1))
        self.pp_field.save = asave

    def test_save(self):
        m = mock.mock_open()
        with mock.patch('__builtin__.open', m, create=True):
            pp.save_fields([self.pp_field], 'foo.pp')
        self.assertTrue(call('foo.pp', 'wb') in m.mock_calls)
        self.assertTrue(call().write('saved') in m.mock_calls)

    def test_save_append(self):
        m = mock.mock_open()
        with mock.patch('__builtin__.open', m, create=True):
            pp.save_fields([self.pp_field], 'foo.pp', append=True)
        self.assertTrue(call('foo.pp', 'ab') in m.mock_calls)
        self.assertTrue(call().write('saved') in m.mock_calls)


if __name__ == "__main__":
    tests.main()
