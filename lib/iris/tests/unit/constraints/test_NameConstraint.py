# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris._constraints.NameConstraint` class."""

import pytest

from iris._constraints import NameConstraint


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.default = "none"

    def test_default(self):
        constraint = NameConstraint()
        assert constraint.standard_name == self.default
        assert constraint.long_name == self.default
        assert constraint.var_name == self.default
        assert constraint.STASH == self.default

    def test_standard_name(self, mocker):
        standard_name = mocker.sentinel.standard_name
        constraint = NameConstraint(standard_name=standard_name)
        assert constraint.standard_name == standard_name
        constraint = NameConstraint(standard_name=standard_name)
        assert constraint.standard_name == standard_name

    def test_long_name(self, mocker):
        long_name = mocker.sentinel.long_name
        constraint = NameConstraint(long_name=long_name)
        assert constraint.standard_name == self.default
        assert constraint.long_name == long_name
        constraint = NameConstraint(standard_name=None, long_name=long_name)
        assert constraint.standard_name is None
        assert constraint.long_name == long_name

    def test_var_name(self, mocker):
        var_name = mocker.sentinel.var_name
        constraint = NameConstraint(var_name=var_name)
        assert constraint.standard_name == self.default
        assert constraint.long_name == self.default
        assert constraint.var_name == var_name
        constraint = NameConstraint(
            standard_name=None, long_name=None, var_name=var_name
        )
        assert constraint.standard_name is None
        assert constraint.long_name is None
        assert constraint.var_name == var_name

    def test_stash(self, mocker):
        STASH = mocker.sentinel.STASH
        constraint = NameConstraint(STASH=STASH)
        assert constraint.standard_name == self.default
        assert constraint.long_name == self.default
        assert constraint.var_name == self.default
        assert constraint.STASH == STASH
        constraint = NameConstraint(
            standard_name=None, long_name=None, var_name=None, STASH=STASH
        )
        assert constraint.standard_name is None
        assert constraint.long_name is None
        assert constraint.var_name is None
        assert constraint.STASH == STASH


class Test__cube_func:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.standard_name = mocker.sentinel.standard_name
        self.long_name = mocker.sentinel.long_name
        self.var_name = mocker.sentinel.var_name
        self.STASH = mocker.sentinel.STASH
        self.cube = mocker.Mock(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            attributes=dict(STASH=self.STASH),
        )

    def test_standard_name(self):
        # Match.
        constraint = NameConstraint(standard_name=self.standard_name)
        assert constraint._cube_func(self.cube)
        # Match.
        constraint = NameConstraint(standard_name=self.standard_name)
        assert constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(standard_name="wibble")
        assert not constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(standard_name="wibble")
        assert not constraint._cube_func(self.cube)

    def test_long_name(self):
        # Match.
        constraint = NameConstraint(long_name=self.long_name)
        assert constraint._cube_func(self.cube)
        # Match.
        constraint = NameConstraint(
            standard_name=self.standard_name, long_name=self.long_name
        )
        assert constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(long_name=None)
        assert not constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(standard_name=None, long_name=self.long_name)
        assert not constraint._cube_func(self.cube)

    def test_var_name(self):
        # Match.
        constraint = NameConstraint(var_name=self.var_name)
        assert constraint._cube_func(self.cube)
        # Match.
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
        )
        assert constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(var_name=None)
        assert not constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(
            standard_name=None, long_name=None, var_name=self.var_name
        )
        assert not constraint._cube_func(self.cube)

    def test_stash(self):
        # Match.
        constraint = NameConstraint(STASH=self.STASH)
        assert constraint._cube_func(self.cube)
        # Match.
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            STASH=self.STASH,
        )
        assert constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(STASH=None)
        assert not constraint._cube_func(self.cube)
        # No match.
        constraint = NameConstraint(
            standard_name=None, long_name=None, var_name=None, STASH=self.STASH
        )
        assert not constraint._cube_func(self.cube)


class Test___repr__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.standard_name = mocker.sentinel.standard_name
        self.long_name = mocker.sentinel.long_name
        self.var_name = mocker.sentinel.var_name
        self.STASH = mocker.sentinel.STASH
        self.msg = "NameConstraint({})"
        self.f_standard_name = "standard_name={!r}".format(self.standard_name)
        self.f_long_name = "long_name={!r}".format(self.long_name)
        self.f_var_name = "var_name={!r}".format(self.var_name)
        self.f_STASH = "STASH={!r}".format(self.STASH)

    def test(self):
        constraint = NameConstraint()
        expected = self.msg.format("")
        assert repr(constraint) == expected

    def test_standard_name(self):
        constraint = NameConstraint(standard_name=self.standard_name)
        expected = self.msg.format(self.f_standard_name)
        assert repr(constraint) == expected

    def test_long_name(self):
        constraint = NameConstraint(long_name=self.long_name)
        expected = self.msg.format(self.f_long_name)
        assert repr(constraint) == expected
        constraint = NameConstraint(
            standard_name=self.standard_name, long_name=self.long_name
        )
        args = "{}, {}".format(self.f_standard_name, self.f_long_name)
        expected = self.msg.format(args)
        assert repr(constraint) == expected

    def test_var_name(self):
        constraint = NameConstraint(var_name=self.var_name)
        expected = self.msg.format(self.f_var_name)
        assert repr(constraint) == expected
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
        )
        args = "{}, {}, {}".format(
            self.f_standard_name, self.f_long_name, self.f_var_name
        )
        expected = self.msg.format(args)
        assert repr(constraint) == expected

    def test_stash(self):
        constraint = NameConstraint(STASH=self.STASH)
        expected = self.msg.format(self.f_STASH)
        assert repr(constraint) == expected
        constraint = NameConstraint(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            STASH=self.STASH,
        )
        args = "{}, {}, {}, {}".format(
            self.f_standard_name,
            self.f_long_name,
            self.f_var_name,
            self.f_STASH,
        )
        expected = self.msg.format(args)
        assert repr(constraint) == expected
