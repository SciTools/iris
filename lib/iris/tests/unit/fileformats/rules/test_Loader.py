# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.rules.Loader`."""

from iris.fileformats.rules import Loader
from iris.tests._shared_utils import assert_no_warnings_regexp


class Test___init__:
    def test_normal(self, mocker):
        with assert_no_warnings_regexp():
            loader = Loader(
                mocker.sentinel.GEN_FUNC,
                mocker.sentinel.GEN_FUNC_KWARGS,
                mocker.sentinel.CONVERTER,
            )
        assert loader.field_generator is mocker.sentinel.GEN_FUNC
        assert loader.field_generator_kwargs is mocker.sentinel.GEN_FUNC_KWARGS
        assert loader.converter is mocker.sentinel.CONVERTER

    def test_normal_with_explicit_none(self, mocker):
        with assert_no_warnings_regexp():
            loader = Loader(
                mocker.sentinel.GEN_FUNC,
                mocker.sentinel.GEN_FUNC_KWARGS,
                mocker.sentinel.CONVERTER,
            )
        assert loader.field_generator is mocker.sentinel.GEN_FUNC
        assert loader.field_generator_kwargs is mocker.sentinel.GEN_FUNC_KWARGS
        assert loader.converter is mocker.sentinel.CONVERTER
