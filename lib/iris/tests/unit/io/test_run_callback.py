# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.io.run_callback` function."""

import pytest

import iris.exceptions
import iris.io


class Test_run_callback:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.cube = mocker.sentinel.cube

    def test_no_callback(self):
        # No callback results in the cube being returned.
        assert iris.io.run_callback(None, self.cube, None, None) == self.cube

    def test_ignore_cube(self):
        # Ignore cube should result in None being returned.
        def callback(cube, field, fname):
            raise iris.exceptions.IgnoreCubeException()

        cube = self.cube
        assert iris.io.run_callback(callback, cube, None, None) is None

    def test_callback_no_return(self):
        # Check that a callback not returning anything still results in the
        # cube being passed back from "run_callback".
        def callback(cube, field, fname):
            pass

        cube = self.cube
        assert iris.io.run_callback(callback, cube, None, None) == cube

    def test_bad_callback_return_type(self):
        # Check that a TypeError is raised with a bad callback return value.
        def callback(cube, field, fname):
            return iris.cube.CubeList()

        emsg = "Callback function returned an unhandled data type."
        with pytest.raises(TypeError, match=emsg):
            iris.io.run_callback(callback, None, None, None)

    def test_bad_signature(self):
        # Check that a TypeError is raised with a bad callback function
        # signature.
        def callback(cube):
            pass

        emsg = "takes 1 positional argument "
        with pytest.raises(TypeError, match=emsg):
            iris.io.run_callback(callback, None, None, None)

    def test_callback_args(self, mocker):
        # Check that the appropriate args are passed through to the callback.
        self.field = mocker.sentinel.field
        self.fname = mocker.sentinel.fname

        def callback(cube, field, fname):
            assert cube == self.cube
            assert field == self.field
            assert fname == self.fname

        iris.io.run_callback(callback, self.cube, self.field, self.fname)
