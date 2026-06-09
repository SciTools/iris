# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris._combine.CombineOptions` class."""

from unittest import mock

import pytest

from iris import CombineOptions


class TestInit:
    def test_init_empty(self):
        # Check how a bare init works
        options = CombineOptions()
        assert options.settings() == CombineOptions.SETTINGS["default"]

    def test_init_args_kwargs(self):
        # Check that init with args, kwargs equates to a pair of set() calls.
        with mock.patch("iris.CombineOptions.set") as mock_set:
            test_option = mock.sentinel.option
            test_kwargs = {"junk": "invalid"}
            CombineOptions(options=test_option, **test_kwargs)
        assert mock_set.call_args_list == [
            mock.call("default"),
            mock.call(test_option, **test_kwargs),
        ]


class Test_settings:
    """The .settings() returns a dict full of the settings."""

    def test_settings(self):
        options = CombineOptions()
        settings = options.settings()
        assert isinstance(settings, dict)
        assert list(settings.keys()) == CombineOptions.OPTION_KEYS
        for key in CombineOptions.OPTION_KEYS:
            assert settings[key] == getattr(options, key)


def options_checks(options, checks):
    # Check (parts of) options against a dictionary of "expected" values.
    settings = options.settings()
    return all(settings[key] == value for key, value in checks.items())


class Test_set_and_context:
    """Check the .set(arg, **kwargs) and .context(arg, **kwargs) behaviours."""

    @staticmethod
    def do_check(
        op_arg=None,
        op_kwargs=None,
        before_checks=None,
        after_checks=None,
        initial_options=None,
        op_is_set=True,
    ):
        """Generic test routine check method.

        Perform an operation(op_arg, **op_kwargs) and test (partial) options state
        before and after.  If provided, can also start from a non-default
        'initial_options' state.

        Used to generalise between the .set() and .context() calls.
        In the case of .context(), the 'after' is within the block, and the 'before'
        state should always be restored again afterwards.
        """
        if initial_options is not None:
            options = initial_options
        else:
            options = CombineOptions()

        op_kwargs = op_kwargs or {}

        if before_checks is not None:
            assert options_checks(options, before_checks)

        if op_is_set:
            # do "set" check
            options.set(op_arg, **op_kwargs)
            assert options_checks(options, after_checks)
        else:
            # do "context" checks
            with options.context(op_arg, **op_kwargs):
                assert options_checks(options, after_checks)
            assert options_checks(options, before_checks)

    @pytest.fixture(params=["set", "context"])
    def op_is_set(self, request):
        """Parametrise a test over both .set() and and .context() calls."""
        return request.param == "set"

    def test_empty_set(self):
        # More or less, just check that an empty set() call is OK.
        options = CombineOptions()
        orig_settings = options.settings()
        options.set()
        assert options.settings() == orig_settings

    def test_empty_context(self):
        # More or less, just check that an empty context() call is OK.
        options = CombineOptions()
        orig_settings = options.settings()
        with options.context():
            assert options.settings() == orig_settings

    def test_arg_dict(self, op_is_set):
        expect_before = {"merge_concat_sequence": "m", "repeat_until_unchanged": False}
        set_arg = {"merge_concat_sequence": "c", "repeat_until_unchanged": True}
        expect_after = {"merge_concat_sequence": "c", "repeat_until_unchanged": True}
        self.do_check(
            op_arg=set_arg,
            before_checks=expect_before,
            after_checks=expect_after,
            op_is_set=op_is_set,
        )

    def test_arg_string(self, op_is_set):
        expect_before = {"merge_concat_sequence": "m", "repeat_until_unchanged": False}
        set_arg = "comprehensive"
        expect_after = {"merge_concat_sequence": "mc", "repeat_until_unchanged": True}
        self.do_check(
            op_arg=set_arg,
            before_checks=expect_before,
            after_checks=expect_after,
            op_is_set=op_is_set,
        )

    def test_kwargs(self, op_is_set):
        expect_before = {"merge_concat_sequence": "m", "repeat_until_unchanged": False}
        set_arg = {"merge_concat_sequence": "c", "repeat_until_unchanged": True}
        expect_after = {"merge_concat_sequence": "c", "repeat_until_unchanged": True}
        self.do_check(
            op_arg=set_arg,
            before_checks=expect_before,
            after_checks=expect_after,
            op_is_set=op_is_set,
        )

    def test_arg_dict_plus_kwargs(self, op_is_set):
        # Show that kwargs override dictionary arg
        expect_before = {"merge_concat_sequence": "m", "repeat_until_unchanged": False}
        # NOTE: the arg changes the sequence from "m" to "c" ...
        set_arg = dict(merge_concat_sequence="c", repeat_until_unchanged=True)
        # .. but the keyword overrides that to "mc"
        set_kwargs = dict(merge_concat_sequence="mc")
        expect_after = {"merge_concat_sequence": "mc", "repeat_until_unchanged": True}
        self.do_check(
            before_checks=expect_before,
            op_arg=set_arg,
            op_kwargs=set_kwargs,
            after_checks=expect_after,
            op_is_set=op_is_set,
        )

    def test_arg_str_plus_kwargs(self, op_is_set):
        # Show that kwargs override settings-name arg
        expect_before = {"merge_concat_sequence": "m", "repeat_until_unchanged": False}
        # NOTE: the arg changes 'sequence' to "mc", and 'repeat' to True ...
        set_arg = "comprehensive"
        # .. but the keyword overrides 'repeat' to False again
        set_kwargs = dict(repeat_until_unchanged=False)
        expect_after = {"merge_concat_sequence": "mc", "repeat_until_unchanged": False}
        self.do_check(
            before_checks=expect_before,
            op_arg=set_arg,
            op_kwargs=set_kwargs,
            after_checks=expect_after,
            op_is_set=op_is_set,
        )

    def test_arg_bad_dict(self):
        options = CombineOptions()
        expected = "Unknown options.*'junk'.* : valid options are"
        with pytest.raises(ValueError, match=expected):
            options.set({"junk": "invalid"})

    def test_arg_bad_string(self):
        options = CombineOptions()
        expected = (
            r"arg 'options'='oddthing'.*not a valid setting.*expected one of.* "
            "['legacy', 'default', 'recommended', 'comprehensive']"
        )
        with pytest.raises(ValueError, match=expected):
            options.set("oddthing")

    def test_bad_kwarg(self):
        options = CombineOptions()
        expected = "Unknown options.*'junk'.* : valid options are"
        with pytest.raises(ValueError, match=expected):
            options.set({"junk": "invalid"})


class Test_AttributeAccess:
    """Check operation of direct property access (with ".")."""

    def test_getattr(self):
        options = CombineOptions(merge_concat_sequence="m")
        assert options.merge_concat_sequence == "m"

    def test_getattr_badname(self):
        options = CombineOptions()
        expected = "'CombineOptions' object has no attribute 'unknown'"
        with pytest.raises(AttributeError, match=expected):
            options.unknown

    def test_setattr(self):
        options = CombineOptions(merge_concat_sequence="m")
        options.merge_concat_sequence = "mc"
        assert options.merge_concat_sequence == "mc"

    def test_setattr_badname(self):
        options = CombineOptions()
        expected = "CombineOptions object has no property 'anyold_property'"
        with pytest.raises(KeyError, match=expected):
            options.anyold_property = "x"

    def test_setattr_badvalue(self):
        options = CombineOptions()
        expected = "'mcm' is not a valid.*merge_concat_sequence : must be one of"
        with pytest.raises(ValueError, match=expected):
            options.merge_concat_sequence = "mcm"
