# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.fileformats.pp_load_rules._model_level_number`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.fileformats.pp_load_rules import _model_level_number


class Test_9999(tests.IrisTest):
    def test(self):
        self.assertEqual(_model_level_number(9999), 0)


class Test_lblev(tests.IrisTest):
    def test(self):
        for val in range(9999):
            self.assertEqual(_model_level_number(val), val)


if __name__ == "__main__":
    tests.main()
