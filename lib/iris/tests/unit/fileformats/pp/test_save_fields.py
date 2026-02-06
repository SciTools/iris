# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp.save_fields` function."""

import numpy as np
import pytest

import iris.fileformats.pp as pp


def asave(afilehandle):
    afilehandle.write("saved")


class TestSaveFields:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        # Create a test object to stand in for a real PPField.
        self.pp_field = mocker.MagicMock(spec=pp.PPField3)
        # Add minimal content required by the pp.save operation.
        self.pp_field.HEADER_DEFN = pp.PPField3.HEADER_DEFN
        self.pp_field.data = np.zeros((1, 1))
        self.pp_field.save = asave

    def test_save(self, mocker):
        open_func = "builtins.open"
        m = mocker.mock_open()
        mocker.patch(open_func, m, create=True)
        pp.save_fields([self.pp_field], "foo.pp")
        assert mocker.call("foo.pp", "wb") in m.mock_calls
        assert mocker.call().write("saved") in m.mock_calls

    def test_save_append(self, mocker):
        open_func = "builtins.open"
        m = mocker.mock_open()
        mocker.patch(open_func, m, create=True)
        pp.save_fields([self.pp_field], "foo.pp", append=True)
        assert mocker.call("foo.pp", "ab") in m.mock_calls
        assert mocker.call().write("saved") in m.mock_calls
