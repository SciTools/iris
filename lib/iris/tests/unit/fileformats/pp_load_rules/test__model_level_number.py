# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.fileformats.pp_load_rules._model_level_number`."""

from iris.fileformats.pp_load_rules import _model_level_number


class Test_9999:
    def test(self):
        assert _model_level_number(9999) == 0


class Test_lblev:
    def test(self):
        for val in range(9999):
            assert _model_level_number(val) == val
