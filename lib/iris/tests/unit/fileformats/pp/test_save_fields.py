# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp.save_fields` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

import iris.fileformats.pp as pp


def asave(afilehandle):
    afilehandle.write("saved")


class TestSaveFields(tests.IrisTest):
    def setUp(self):
        # Create a test object to stand in for a real PPField.
        self.pp_field = mock.MagicMock(spec=pp.PPField3)
        # Add minimal content required by the pp.save operation.
        self.pp_field.HEADER_DEFN = pp.PPField3.HEADER_DEFN
        self.pp_field.data = np.zeros((1, 1))
        self.pp_field.save = asave

    def test_save(self):
        open_func = "builtins.open"
        m = mock.mock_open()
        with mock.patch(open_func, m, create=True):
            pp.save_fields([self.pp_field], "foo.pp")
        self.assertTrue(mock.call("foo.pp", "wb") in m.mock_calls)
        self.assertTrue(mock.call().write("saved") in m.mock_calls)

    def test_save_append(self):
        open_func = "builtins.open"
        m = mock.mock_open()
        with mock.patch(open_func, m, create=True):
            pp.save_fields([self.pp_field], "foo.pp", append=True)
        self.assertTrue(mock.call("foo.pp", "ab") in m.mock_calls)
        self.assertTrue(mock.call().write("saved") in m.mock_calls)


if __name__ == "__main__":
    tests.main()
