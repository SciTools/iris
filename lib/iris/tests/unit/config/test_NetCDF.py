# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.config.NetCDF` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

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
        bad_value = "wibble"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.options.conventions_override = bad_value
        self.assertFalse(self.options.conventions_override)
        exp_wmsg = "Attempting to set invalid value {!r}".format(bad_value)
        self.assertRegex(str(w[0].message), exp_wmsg)

    def test__contextmgr(self):
        with self.options.context(conventions_override=True):
            self.assertTrue(self.options.conventions_override)
        self.assertFalse(self.options.conventions_override)


if __name__ == "__main__":
    tests.main()
