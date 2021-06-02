# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris.fileformats.rules.Loader`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.fileformats.rules import Loader


class Test___init__(tests.IrisTest):
    def test_normal(self):
        with mock.patch("warnings.warn") as warn:
            loader = Loader(
                mock.sentinel.GEN_FUNC,
                mock.sentinel.GEN_FUNC_KWARGS,
                mock.sentinel.CONVERTER,
            )
        self.assertEqual(warn.call_count, 0)
        self.assertIs(loader.field_generator, mock.sentinel.GEN_FUNC)
        self.assertIs(
            loader.field_generator_kwargs, mock.sentinel.GEN_FUNC_KWARGS
        )
        self.assertIs(loader.converter, mock.sentinel.CONVERTER)

    def test_normal_with_explicit_none(self):
        with mock.patch("warnings.warn") as warn:
            loader = Loader(
                mock.sentinel.GEN_FUNC,
                mock.sentinel.GEN_FUNC_KWARGS,
                mock.sentinel.CONVERTER,
            )
        self.assertEqual(warn.call_count, 0)
        self.assertIs(loader.field_generator, mock.sentinel.GEN_FUNC)
        self.assertIs(
            loader.field_generator_kwargs, mock.sentinel.GEN_FUNC_KWARGS
        )
        self.assertIs(loader.converter, mock.sentinel.CONVERTER)


if __name__ == "__main__":
    tests.main()
