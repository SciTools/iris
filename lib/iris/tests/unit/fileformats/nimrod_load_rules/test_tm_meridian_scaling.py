# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `iris.fileformats.nimrod_load_rules.tm_meridian_scaling`
function.

"""

from __future__ import absolute_import, division, print_function
from six.moves import filter, input, map, range, zip  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

from iris.fileformats.nimrod_load_rules import (
    tm_meridian_scaling,
    NIMROD_DEFAULT,
    MERIDIAN_SCALING_BNG,
)
from iris.fileformats.nimrod import NimrodField


class Test(tests.IrisTest):
    def setUp(self):
        self.field = mock.Mock(
            tm_meridian_scaling=NIMROD_DEFAULT,
            spec=NimrodField,
            float32_mdi=-123,
        )
        self.cube = mock.Mock()

    def _call_tm_meridian_scaling(self, scaling_value):
        self.field.tm_meridian_scaling = scaling_value
        tm_meridian_scaling(self.cube, self.field)

    def test_unhandled(self):
        with mock.patch("warnings.warn") as warn:
            self._call_tm_meridian_scaling(1)
        self.assertEqual(warn.call_count, 1)

    @tests.no_warnings
    def test_british_national_grid(self):
        # A value is not returned in this rule currently.
        self.assertEqual(
            None, self._call_tm_meridian_scaling(MERIDIAN_SCALING_BNG)
        )

    def test_null(self):
        with mock.patch("warnings.warn") as warn:
            self._call_tm_meridian_scaling(NIMROD_DEFAULT)
            self._call_tm_meridian_scaling(self.field.float32_mdi)
        self.assertEqual(warn.call_count, 0)


if __name__ == "__main__":
    tests.main()
