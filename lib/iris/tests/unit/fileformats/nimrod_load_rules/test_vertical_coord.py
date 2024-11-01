# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.nimrod_load_rules.vertical_coord`
function.

"""

import pytest

from iris.fileformats.nimrod import NimrodField
from iris.fileformats.nimrod_load_rules import (
    NIMROD_DEFAULT,
    TranslationWarning,
    vertical_coord,
)
from iris.tests._shared_utils import assert_no_warnings_regexp


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.field = mocker.Mock(
            vertical_coord=NIMROD_DEFAULT,
            vertical_coord_type=NIMROD_DEFAULT,
            reference_vertical_coord=NIMROD_DEFAULT,
            reference_vertical_coord_type=NIMROD_DEFAULT,
            int_mdi=-32767,
            float32_mdi=NIMROD_DEFAULT,
            spec=NimrodField,
        )
        self.cube = mocker.Mock()

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
            self.field.reference_vertical_coord_type = reference_vertical_coord_type
        vertical_coord(self.cube, self.field)

    def test_unhandled(self):
        message_regexp = "Vertical coord -1 not yet handled"
        with pytest.warns(TranslationWarning, match=message_regexp):
            self._call_vertical_coord(vertical_coord_val=1.0, vertical_coord_type=-1)

    def test_null(self):
        with assert_no_warnings_regexp():
            self._call_vertical_coord(vertical_coord_type=NIMROD_DEFAULT)
            self._call_vertical_coord(vertical_coord_type=self.field.int_mdi)

    def test_ground_level(self):
        with assert_no_warnings_regexp():
            self._call_vertical_coord(vertical_coord_val=9999.0, vertical_coord_type=0)
