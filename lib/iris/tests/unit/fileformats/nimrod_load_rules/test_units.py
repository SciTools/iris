# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.nimrod_load_rules.units` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.cube import Cube
from iris.fileformats.nimrod import NimrodField
from iris.fileformats.nimrod_load_rules import NIMROD_DEFAULT, units


class Test(tests.IrisTest):
    NIMROD_LOCATION = "iris.fileformats.nimrod_load_rules"

    def setUp(self):
        self.field = mock.Mock(
            units="",
            int_mdi=-32767,
            float32_mdi=NIMROD_DEFAULT,
            spec=NimrodField,
        )
        self.cube = Cube(np.ones((3, 3), dtype=np.float32))

    def _call_units(self, data=None, units_str=None):
        if data is not None:
            self.cube.data = data
        if units_str:
            self.field.units = units_str
        units(self.cube, self.field)

    def test_null(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(units_str="m")
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "m")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))

    def test_times32(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(
                data=np.ones_like(self.cube.data) * 32, units_str="mm/hr*32"
            )
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "mm/hr")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_visibility_units(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(
                data=((np.ones_like(self.cube.data) / 2) - 25000),
                units_str="m/2-25k",
            )
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "m")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_power_in_units(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(
                data=np.ones_like(self.cube.data) * 1000, units_str="mm*10^3"
            )
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "mm")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_ug_per_m3_units(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(
                data=(np.ones_like(self.cube.data) * 10),
                units_str="ug/m3E1",
            )
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "ug/m3")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_g_per_kg(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(
                data=(np.ones_like(self.cube.data) * 1000), units_str="g/Kg"
            )
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "kg/kg")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_unit_expection_dictionary(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(units_str="mb")
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "hPa")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_per_second(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(units_str="/s")
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(self.cube.units, "s^-1")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_unhandled_unit(self):
        with mock.patch("warnings.warn") as warn:
            self._call_units(units_str="kittens")
        self.assertEqual(warn.call_count, 1)
        self.assertEqual(self.cube.units, "")
        self.assertArrayAlmostEqual(self.cube.data, np.ones_like(self.cube.data))
        self.assertEqual(self.cube.data.dtype, np.float32)
        self.assertEqual(self.cube.attributes["invalid_units"], "kittens")


if __name__ == "__main__":
    tests.main()
