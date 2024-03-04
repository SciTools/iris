# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.abf.ABFField` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.fileformats.abf import ABFField


class MethodCounter:
    def __init__(self, method_name):
        self.method_name = method_name
        self.count = 0

    def __enter__(self):
        self.orig_method = getattr(ABFField, self.method_name)

        def new_method(*args, **kwargs):
            self.count += 1
            self.orig_method(*args, **kwargs)

        setattr(ABFField, self.method_name, new_method)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(ABFField, self.method_name, self.orig_method)
        return False


class Test_data(tests.IrisTest):
    def test_single_read(self):
        path = "0000000000000000jan00000"
        field = ABFField(path)

        with mock.patch("iris.fileformats.abf.np.fromfile") as fromfile:
            with MethodCounter("__getattr__") as getattr:
                with MethodCounter("_read") as read:
                    field.data

        fromfile.assert_called_once_with(path, dtype=">u1")
        self.assertEqual(getattr.count, 1)
        self.assertEqual(read.count, 1)


if __name__ == "__main__":
    tests.main()
