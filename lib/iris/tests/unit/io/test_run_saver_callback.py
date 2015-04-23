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
"""Unit tests for the `iris.io.run_saver_callback` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris
from iris.exceptions import IgnoreFieldException
from iris.io import run_saver_callback


class Test(tests.IrisTest):
    def setUp(self):
        self.cube = mock.sentinel.cube
        self.field = mock.sentinel.field
        self.fname = mock.sentinel.fname

    def test_no_callback(self):
        # No callback results in the field being returned.
        result = run_saver_callback(None, None, self.field, None)
        self.assertEqual(result, self.field)

    def test_ignore_field(self):
        # Ignore field should result in None being returned.
        def callback(cube, field, fname):
            raise IgnoreFieldException()
        result = run_saver_callback(callback, None, self.field, None)
        self.assertIsNone(result)

    def test_callback_no_return(self):
        # Check that a callback not returning anything still results in the
        # cube being passed back from "run_saver_callback".
        def callback(cube, field, fname):
            pass
        result = run_saver_callback(callback, None, self.field, None)
        self.assertEqual(result, self.field)

    def test_bad_callback_return_type(self):
        # Check that a TypeError is raised with a bad callback return value.
        def callback(cube, field, fname):
            return iris.cube.CubeList()
        emsg = 'Saver callback function returned an unhandled data type'
        with self.assertRaisesRegexp(TypeError, emsg):
            run_saver_callback(callback, None, None, None)

    def test_bad_signature(self):
        # Check that a TypeError is raised with a bad callback function
        # signature.
        def callback(cube):
            pass
        emsg = 'takes exactly 1 argument'
        with self.assertRaisesRegexp(TypeError, emsg):
            run_saver_callback(callback, None, None, None)

    def test_callback_args(self):
        # Check that the appropriate args are passed through to the callback.
        def callback(cube, field, fname):
            self.assertEqual(cube, self.cube)
            self.assertEqual(field, self.field)
            self.assertEqual(fname, self.fname)
        run_saver_callback(callback, self.cube, self.field, self.fname)


if __name__ == "__main__":
    tests.main()
