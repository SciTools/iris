# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `iris.fileformats.nimrod_load_rules.units` function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock
import numpy as np

from iris.cube import Cube
from iris.fileformats.nimrod_load_rules import (
    units,
    NIMROD_DEFAULT,
)
from iris.fileformats.nimrod import NimrodField


class Test(tests.IrisTest):
    NIMROD_LOCATION = "iris.fileformats.nimrod_load_rules"

    def setUp(self):
        self.field = mock.Mock(
            units="",
            int_mdi=-32767,
            float32_mdi=NIMROD_DEFAULT,
            spec=NimrodField,
        )
        self.cube = mock.Mock(
            data=np.zeros((3, 3), dtype=np.float32), spec=Cube
        )

    def _call_units(
        self, data=None, units_str=None,
    ):
        if data is not None:
            self.cube.data = data
        if units_str:
            self.field.units = units_str
        units(self.cube, self.field)

    def test_null(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(data=np.ones_like(self.cube.data), units_str="m")
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "m")
        self.assertArrayAlmostEqual(
            self.cube.data, np.ones_like(self.cube.data)
        )

    def test_times32(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(
                data=np.ones_like(self.cube.data) * 32, units_str="mm/hr*32"
            )
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "mm/hr")
        self.assertArrayAlmostEqual(
            self.cube.data, np.ones_like(self.cube.data)
        )


if __name__ == "__main__":
    tests.main()
