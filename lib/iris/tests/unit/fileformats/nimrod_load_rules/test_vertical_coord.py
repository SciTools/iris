# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `iris.fileformats.nimrod_load_rules.vertical_coord`
function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

from iris.fileformats.nimrod_load_rules import (
    vertical_coord,
    NIMROD_DEFAULT,
    TranslationWarning,
)
from iris.fileformats.nimrod import NimrodField


class Test(tests.IrisTest):
    NIMROD_LOCATION = "iris.fileformats.nimrod_load_rules"

    def setUp(self):
        self.field = mock.Mock(
            vertical_coord_type=NIMROD_DEFAULT,
            int_mdi=mock.sentinel.int_mdi,
            field_code=mock.sentinel.field_code,
            vertical_coord=mock.sentinel.vertical_coord,
            reference_vertical_coord=mock.sentinel.reference_vertical_coord,
            ensemble_member=NIMROD_DEFAULT,
            spec=NimrodField,
        )
        self.cube = mock.Mock()

    def _call_vertical_coord(self, vertical_coord_type):
        self.field.vertical_coord_type = vertical_coord_type
        vertical_coord(self.cube, self.field)

    def test_unhandled(self):
        with mock.patch("warnings.warn") as warn:
            self._call_vertical_coord(-1)
        warn.assert_called_once_with(
            "Vertical coord -1 not yet handled", TranslationWarning
        )

    def test_height(self):
        name = "vertical_coord"
        with mock.patch(self.NIMROD_LOCATION + "." + name) as height:
            self._call_vertical_coord(0)
        height.assert_called_once_with(self.cube, self.field)

    def test_null(self):
        with mock.patch("warnings.warn") as warn:
            self._call_vertical_coord(NIMROD_DEFAULT)
            self._call_vertical_coord(self.field.int_mdi)
        self.assertEqual(warn.call_count, 0)


if __name__ == "__main__":
    tests.main()
