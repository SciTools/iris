# (C) British Crown Copyright 2013 - 2019, Met Office
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
"""Unit tests for the `iris.Future` class."""

from __future__ import (absolute_import, division, print_function)
import six
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
from unittest import skip

import warnings

from iris import Future


def patched_future(value=True, error=False):
    # This ensures that there exists a flag in Future to test against.
    future = Future()
    future.__dict__['example_future_flag'] = value
    if error:
        future.deprecated_options['example_future_flag'] = 'error'
    else:
        future.deprecated_options['example_future_flag'] = 'warning'
    return future


class Test___setattr__(tests.IrisTest):
    @skip("This flag has been removed. Test kept as a future template.")
    def test_valid_clip_latitudes(self):
        future = Future()
        new_value = not future.clip_latitudes
        msg = "'Future' property 'clip_latitudes' is deprecated"
        with self.assertWarnsRegexp(msg):
            future.clip_latitudes = new_value
        self.assertEqual(future.clip_latitudes, new_value)

    def test_deprecated_warning(self):
        future = patched_future()
        new_value = not future.example_future_flag
        msg = "'Future' property 'example_future_flag' is deprecated"
        with self.assertWarnsRegexp(msg):
            future.example_future_flag = new_value
        self.assertEqual(future.example_future_flag, new_value)

    def test_deprecated_error_warning(self):
        future = patched_future(error=True)
        exp_emsg = "'Future' property 'example_future_flag' is deprecated"
        with self.assertWarnsRegexp(exp_emsg):
            future.example_future_flag = True

    def test_invalid_deprecated_error(self):
        future = patched_future(error=True)
        exp_emsg = \
            "'Future' property 'example_future_flag' has been deprecated"
        with self.assertRaisesRegexp(AttributeError, exp_emsg):
            future.example_future_flag = False

    def test_invalid_attribute(self):
        future = Future()
        with self.assertRaises(AttributeError):
            future.numberwang = 7

    @skip("This flag has been removed. Test kept as a future template.")
    def test_netcdf_no_unlimited(self):
        future = Future()
        exp_emsg = "'Future' property 'netcdf_no_unlimited' is deprecated"
        with self.assertWarnsRegexp(exp_emsg):
            future.netcdf_no_unlimited = True

    @skip("This flag has been removed. Test kept as a future template.")
    def test_invalid_netcdf_no_unlimited(self):
        future = Future()
        exp_emsg = \
            "'Future' property 'netcdf_no_unlimited' has been deprecated"
        with self.assertRaisesRegexp(AttributeError, exp_emsg):
            future.netcdf_no_unlimited = False

    @skip("This flag has been removed. Test kept as a future template.")
    def test_cell_datetime_objects(self):
        future = Future()
        new_value = not future.cell_datetime_objects
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            future.cell_datetime_objects = new_value
        self.assertEqual(future.cell_datetime_objects, new_value)
        exp_wmsg = "'Future' property 'cell_datetime_objects' is deprecated"
        six.assertRegex(self, str(warn[0]), exp_wmsg)


class Test_context(tests.IrisTest):
    # Many of these tests will not work until another flag is put into Future.
    @skip("This flag has been removed. Test kept as a future template.")
    def test_no_args(self):
        # Catch the deprecation when explicitly setting `cell_datetime_objects`
        # as the test is still useful even though the Future property is
        # deprecated.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            future = Future(cell_datetime_objects=False)
            self.assertFalse(future.cell_datetime_objects)
            with future.context():
                self.assertFalse(future.cell_datetime_objects)
                future.cell_datetime_objects = True
                self.assertTrue(future.cell_datetime_objects)
            self.assertFalse(future.cell_datetime_objects)

    def test_generic_no_args(self):
        # While Future has no properties, it is necessary to patch Future in
        # order for these tests to work. This test is not a precise emulation
        # of the test it is replacing, but ought to cover most of the same
        # behaviour while Future is empty.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            future = patched_future(value=False)
            self.assertFalse(future.example_future_flag)
            with future.context():
                self.assertFalse(future.example_future_flag)
                future.example_future_flag = True
                self.assertTrue(future.example_future_flag)
            self.assertFalse(future.example_future_flag)

    @skip("This flag has been removed. Test kept as a future template.")
    def test_with_arg(self):
        # Catch the deprecation when explicitly setting `cell_datetime_objects`
        # as the test is still useful even though the Future property is
        # deprecated.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            future = Future(cell_datetime_objects=False)
            self.assertFalse(future.cell_datetime_objects)
            with future.context(cell_datetime_objects=True):
                self.assertTrue(future.cell_datetime_objects)
            self.assertFalse(future.cell_datetime_objects)

    def test_generic_with_arg(self):
        # While Future has no properties, it is necessary to patch Future in
        # order for these tests to work. This test is not a precise emulation
        # of the test it is replacing, but ought to cover most of the same
        # behaviour while Future is empty.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            future = patched_future(value=False)
            self.assertFalse(future.example_future_flag)
            self.assertFalse(future.example_future_flag)
            with future.context(example_future_flag=True):
                self.assertTrue(future.example_future_flag)
            self.assertFalse(future.example_future_flag)

    def test_invalid_arg(self):
        future = Future()
        with self.assertRaises(AttributeError):
            with future.context(this_does_not_exist=True):
                # Don't need to do anything here... the context manager
                # will (assuming it's working!) have already raised the
                # exception we're looking for.
                pass

    @skip("This flag has been removed. Test kept as a future template.")
    def test_exception(self):
        # Check that an interrupted context block restores the initial state.
        class LocalTestException(Exception):
            pass

        # Catch the deprecation when explicitly setting `cell_datetime_objects`
        # as the test is still useful even though the Future property is
        # deprecated.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            future = Future(cell_datetime_objects=False)
            try:
                with future.context(cell_datetime_objects=True):
                    raise LocalTestException()
            except LocalTestException:
                pass
            self.assertEqual(future.cell_datetime_objects, False)

    def test_generic_exception(self):
        # Check that an interrupted context block restores the initial state.
        class LocalTestException(Exception):
            pass

        # While Future has no properties, it is necessary to patch Future in
        # order for these tests to work. This test is not a precise emulation
        # of the test it is replacing, but ought to cover most of the same
        # behaviour while Future is empty.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            future = patched_future(value=False)
            try:
                with future.context(example_future_flag=True):
                    raise LocalTestException()
            except LocalTestException:
                pass
            self.assertEqual(future.example_future_flag, False)


if __name__ == "__main__":
    tests.main()
