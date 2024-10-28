# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.io.loading.LoadPolicy` package."""

from unittest import mock

import pytest

from iris import LoadPolicy


class TestInit:
    def test_init_empty(self):
        # Check how a bare init works
        options = LoadPolicy()
        assert options.settings() == LoadPolicy.SETTINGS["default"]

    def test_init_args_kwargs(self):
        # Check that init with args, kwargs equates to a pair of set() calls.
        with mock.patch("iris.LoadPolicy.set") as mock_set:
            test_option = mock.sentinel.option
            test_kwargs = {"junk": "invalid"}
            LoadPolicy(options=test_option, **test_kwargs)
        assert mock_set.call_args_list == [
            mock.call("default"),
            mock.call(test_option, **test_kwargs),
        ]


class Test_settings:
    """The .settings() returns a dict full of the settings."""

    def test_settings(self):
        options = LoadPolicy()
        settings = options.settings()
        assert isinstance(settings, dict)
        assert tuple(settings.keys()) == LoadPolicy.OPTION_KEYS
        for key in LoadPolicy.OPTION_KEYS:
            assert settings[key] == getattr(options, key)


class Test_set:
    """Check the .set(arg, **kwargs) behaviour."""

    def test_empty(self):
        options = LoadPolicy()
        orig_settings = options.settings()
        options.set()
        assert options.settings() == orig_settings

    def test_arg_dict(self):
        options = LoadPolicy()
        assert options.settings()["merge_concat_sequence"] == "m"
        assert options.settings()["repeat_until_unchanged"] is False
        options.set({"merge_concat_sequence": "c", "repeat_until_unchanged": True})
        assert options.settings()["merge_concat_sequence"] == "c"
        assert options.settings()["repeat_until_unchanged"] is True

    def test_arg_string(self):
        options = LoadPolicy()
        assert options.settings()["merge_concat_sequence"] == "m"
        assert options.settings()["repeat_until_unchanged"] is False
        options.set("comprehensive")
        assert options.settings()["merge_concat_sequence"] == "mc"
        assert options.settings()["repeat_until_unchanged"] is True

    def test_arg_bad_dict(self):
        options = LoadPolicy()
        expected = "Unknown options.*'junk'.* : valid options are"
        with pytest.raises(ValueError, match=expected):
            options.set({"junk": "invalid"})

    def test_arg_bad_string(self):
        options = LoadPolicy()
        expected = "Invalid arg options='unknown' : must be a dict, or one of"
        with pytest.raises(TypeError, match=expected):
            options.set("unknown")

    def test_arg_bad_type(self):
        options = LoadPolicy()
        expected = "must be a dict, or one of"
        with pytest.raises(TypeError, match=expected):
            options.set((1, 2, 3))

    def test_kwargs(self):
        options = LoadPolicy()
        assert options.settings()["merge_concat_sequence"] == "m"
        assert options.settings()["repeat_until_unchanged"] is False
        options.set(merge_concat_sequence="c", repeat_until_unchanged=True)
        assert options.settings()["merge_concat_sequence"] == "c"
        assert options.settings()["repeat_until_unchanged"] is True

    def test_arg_kwargs(self):
        # Show that kwargs override arg
        options = LoadPolicy(
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
        options = LoadPolicy()
        expected = "Unknown options.*'junk'.* : valid options are"
        with pytest.raises(ValueError, match=expected):
            options.set({"junk": "invalid"})


class Test_AttributeAccess:
    """Check operation of direct property access (with ".")."""

    def test_getattr(self):
        options = LoadPolicy(merge_concat_sequence="m")
        assert options.merge_concat_sequence == "m"

    def test_getattr_badname(self):
        options = LoadPolicy()
        expected = "'LoadPolicy' object has no attribute 'unknown'"
        with pytest.raises(AttributeError, match=expected):
            options.unknown

    def test_setattr(self):
        options = LoadPolicy(merge_concat_sequence="m")
        options.merge_concat_sequence = "mc"
        assert options.merge_concat_sequence == "mc"

    def test_setattr_badname(self):
        options = LoadPolicy()
        expected = "LoadPolicy object has no property 'anyold_property'"
        with pytest.raises(KeyError, match=expected):
            options.anyold_property = "x"

    def test_setattr_badvalue(self):
        options = LoadPolicy()
        expected = "'mcm' is not a valid.*merge_concat_sequence : must be one of"
        with pytest.raises(ValueError, match=expected):
            options.merge_concat_sequence = "mcm"
