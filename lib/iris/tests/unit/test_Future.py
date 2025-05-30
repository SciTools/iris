# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.Future` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import warnings

from iris import Future
import iris._deprecation


def patched_future(value=False, deprecated=False, error=False):
    class LocalFuture(Future):
        # Modified Future class, with controlled deprecation options.
        #
        # NOTE: it is necessary to subclass this in order to modify the
        # 'deprecated_options' property, because we don't want to modify the
        # class variable of the actual Future class !
        deprecated_options = {}
        if deprecated:
            if error:
                deprecated_options["example_future_flag"] = "error"
            else:
                deprecated_options["example_future_flag"] = "warning"

    future = LocalFuture()
    future.__dict__["example_future_flag"] = value
    return future


class Test___setattr__(tests.IrisTest):
    def test_valid_setting(self):
        future = patched_future()
        new_value = not future.example_future_flag
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Check no warning emitted !
            future.example_future_flag = new_value
        self.assertEqual(future.example_future_flag, new_value)

    def test_deprecated_warning(self):
        future = patched_future(deprecated=True, error=False)
        msg = "'Future' property 'example_future_flag' is deprecated"
        with self.assertWarnsRegex(iris._deprecation.IrisDeprecation, msg):
            future.example_future_flag = False

    def test_deprecated_error(self):
        future = patched_future(deprecated=True, error=True)
        exp_emsg = "'Future' property 'example_future_flag' has been deprecated"
        with self.assertRaisesRegex(AttributeError, exp_emsg):
            future.example_future_flag = False

    def test_invalid_attribute(self):
        future = Future()
        with self.assertRaises(AttributeError):
            future.numberwang = 7


class Test_context(tests.IrisTest):
    def test_generic_no_args(self):
        # While Future has no properties, it is necessary to patch Future in
        # order for these tests to work. This test is not a precise emulation
        # of the test it is replacing, but ought to cover most of the same
        # behaviour while Future is empty.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            future = patched_future(value=False)
            self.assertFalse(future.example_future_flag)
            with future.context():
                self.assertFalse(future.example_future_flag)
                future.example_future_flag = True
                self.assertTrue(future.example_future_flag)
            self.assertFalse(future.example_future_flag)

    def test_generic_with_arg(self):
        # While Future has no properties, it is necessary to patch Future in
        # order for these tests to work. This test is not a precise emulation
        # of the test it is replacing, but ought to cover most of the same
        # behaviour while Future is empty.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

    def test_generic_exception(self):
        # Check that an interrupted context block restores the initial state.
        class LocalTestException(Exception):
            pass

        # While Future has no properties, it is necessary to patch Future in
        # order for these tests to work. This test is not a precise emulation
        # of the test it is replacing, but ought to cover most of the same
        # behaviour while Future is empty.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            future = patched_future(value=False)
            try:
                with future.context(example_future_flag=True):
                    raise LocalTestException()
            except LocalTestException:
                pass
            self.assertEqual(future.example_future_flag, False)


class Test__str_repr:
    # Very basic check on printed forms of the Future object
    def _check_content(self, future, text):
        assert text == (
            "Future(datum_support=False, pandas_ndim=False, save_split_attrs=False, "
            "date_microseconds=False, derived_bounds=False)"
        )
        # Also just check that all the property elements are included
        for propname in future.__dict__.keys():
            assert f"{propname}=False" in text

    def test_str(self):
        future = Future()
        text = str(future)
        self._check_content(future, text)

    def test_repr(self):
        future = Future()
        text = repr(future)
        self._check_content(future, text)


if __name__ == "__main__":
    tests.main()
