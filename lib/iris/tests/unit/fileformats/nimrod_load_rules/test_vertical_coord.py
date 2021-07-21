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
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.fileformats.nimrod import NimrodField
from iris.fileformats.nimrod_load_rules import (
    NIMROD_DEFAULT,
    TranslationWarning,
    vertical_coord,
)


class Test(tests.IrisTest):
    NIMROD_LOCATION = "iris.fileformats.nimrod_load_rules"

    def setUp(self):
        self.field = mock.Mock(
            vertical_coord=NIMROD_DEFAULT,
            vertical_coord_type=NIMROD_DEFAULT,
            reference_vertical_coord=NIMROD_DEFAULT,
            reference_vertical_coord_type=NIMROD_DEFAULT,
            int_mdi=-32767,
            float32_mdi=NIMROD_DEFAULT,
            spec=NimrodField,
        )
        self.cube = mock.Mock()

    def _call_vertical_coord(
        self,
        vertical_coord_val=None,
        vertical_coord_type=None,
        reference_vertical_coord=None,
        reference_vertical_coord_type=None,
    ):
        if vertical_coord_val:
            self.field.vertical_coord = vertical_coord_val
        if vertical_coord_type:
            self.field.vertical_coord_type = vertical_coord_type
        if reference_vertical_coord:
            self.field.reference_vertical_coord = reference_vertical_coord
        if reference_vertical_coord_type:
            self.field.reference_vertical_coord_type = (
                reference_vertical_coord_type
            )
        vertical_coord(self.cube, self.field)

    def test_unhandled(self):
        with mock.patch("warnings.warn") as warn:
            self._call_vertical_coord(
                vertical_coord_val=1.0, vertical_coord_type=-1
            )
        warn.assert_called_once_with(
            "Vertical coord -1 not yet handled", TranslationWarning
        )

    def test_null(self):
        with mock.patch("warnings.warn") as warn:
            self._call_vertical_coord(vertical_coord_type=NIMROD_DEFAULT)
            self._call_vertical_coord(vertical_coord_type=self.field.int_mdi)
        self.assertEqual(warn.call_count, 0)

    def test_ground_level(self):
        with mock.patch("warnings.warn") as warn:
            self._call_vertical_coord(
                vertical_coord_val=9999.0, vertical_coord_type=0
            )
        self.assertEqual(warn.call_count, 0)


if __name__ == "__main__":
    tests.main()
