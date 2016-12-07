# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Unit tests for the `iris.io.run_callback` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.exceptions
import iris.io
from iris.tests import mock


class Test_run_callback(tests.IrisTest):
    def setUp(self):
        tests.IrisTest.setUp(self)
        self.cube = mock.sentinel.cube

    def test_no_callback(self):
        # No callback results in the cube being returned.
        self.assertEqual(iris.io.run_callback(None, self.cube, None, None),
                         self.cube)

    def test_ignore_cube(self):
        # Ignore cube should result in None being returned.
        def callback(cube, field, fname):
            raise iris.exceptions.IgnoreCubeException()
        cube = self.cube
        self.assertEqual(iris.io.run_callback(callback, cube, None, None),
                         None)

    def test_callback_no_return(self):
        # Check that a callback not returning anything still results in the
        # cube being passed back from "run_callback".
        def callback(cube, field, fname):
            pass

        cube = self.cube
        self.assertEqual(iris.io.run_callback(callback, cube, None, None),
                         cube)

    def test_bad_callback_return_type(self):
        # Check that a TypeError is raised with a bad callback return value.
        def callback(cube, field, fname):
            return iris.cube.CubeList()
        with self.assertRaisesRegexp(TypeError,
                                     'Callback function returned an '
                                     'unhandled data type.'):
            iris.io.run_callback(callback, None, None, None)

    def test_bad_signature(self):
        # Check that a TypeError is raised with a bad callback function
        # signature.
        def callback(cube):
            pass
        with self.assertRaisesRegexp(TypeError,
                                     # exactly == Py2, positional == Py3
                                     'takes (exactly )?1 (positional )?'
                                     'argument '):
            iris.io.run_callback(callback, None, None, None)

    def test_callback_args(self):
        # Check that the appropriate args are passed through to the callback.
        self.field = mock.sentinel.field
        self.fname = mock.sentinel.fname

        def callback(cube, field, fname):
            self.assertEqual(cube, self.cube)
            self.assertEqual(field, self.field)
            self.assertEqual(fname, self.fname)

        iris.io.run_callback(callback, self.cube, self.field, self.fname)


if __name__ == "__main__":
    tests.main()
