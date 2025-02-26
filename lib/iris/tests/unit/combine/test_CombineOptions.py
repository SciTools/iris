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


class Test_set:
    """Check the .set(arg, **kwargs) behaviour."""

    def test_empty(self):
        options = CombineOptions()
        orig_settings = options.settings()
        options.set()
        assert options.settings() == orig_settings

    def test_arg_dict(self):
        options = CombineOptions()
        assert options.settings()["merge_concat_sequence"] == "m"
        assert options.settings()["repeat_until_unchanged"] is False
        options.set({"merge_concat_sequence": "c", "repeat_until_unchanged": True})
        assert options.settings()["merge_concat_sequence"] == "c"
        assert options.settings()["repeat_until_unchanged"] is True

    def test_arg_string(self):
        options = CombineOptions()
        assert options.settings()["merge_concat_sequence"] == "m"
        assert options.settings()["repeat_until_unchanged"] is False
        options.set("comprehensive")
        assert options.settings()["merge_concat_sequence"] == "mc"
        assert options.settings()["repeat_until_unchanged"] is True

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

    def test_kwargs(self):
        options = CombineOptions()
        assert options.settings()["merge_concat_sequence"] == "m"
        assert options.settings()["repeat_until_unchanged"] is False
        options.set(merge_concat_sequence="c", repeat_until_unchanged=True)
        assert options.settings()["merge_concat_sequence"] == "c"
        assert options.settings()["repeat_until_unchanged"] is True

    def test_arg_kwargs(self):
        # Show that kwargs override arg
        options = CombineOptions(
            support_multiple_references=False,
            merge_concat_sequence="",
            repeat_until_unchanged=False,
        )
        options.set(
            dict(merge_concat_sequence="c", repeat_until_unchanged=True),
            merge_concat_sequence="mc",
        )
        assert options.merge_concat_sequence == "mc"
        assert options.repeat_until_unchanged is True

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
