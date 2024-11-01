# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.nimrod_load_rules.units` function."""

import numpy as np
import pytest

from iris.cube import Cube
from iris.fileformats.nimrod import NimrodField
from iris.fileformats.nimrod_load_rules import NIMROD_DEFAULT, units
from iris.tests._shared_utils import (
    assert_array_almost_equal,
    assert_no_warnings_regexp,
)


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.field = mocker.Mock(
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

    def test_null(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(units_str="m")
        assert self.cube.units == "m"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))

    def test_times32(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(
                data=np.ones_like(self.cube.data) * 32, units_str="mm/hr*32"
            )
        assert self.cube.units == "mm/hr"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32

    def test_visibility_units(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(
                data=((np.ones_like(self.cube.data) / 2) - 25000),
                units_str="m/2-25k",
            )
        assert self.cube.units == "m"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32

    def test_power_in_units(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(
                data=np.ones_like(self.cube.data) * 1000, units_str="mm*10^3"
            )
        assert self.cube.units == "mm"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32

    def test_ug_per_m3_units(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(
                data=(np.ones_like(self.cube.data) * 10),
                units_str="ug/m3E1",
            )
        assert self.cube.units == "ug/m3"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32

    def test_g_per_kg(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(
                data=(np.ones_like(self.cube.data) * 1000), units_str="g/Kg"
            )
        assert self.cube.units == "kg/kg"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32

    def test_unit_expection_dictionary(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(units_str="mb")
        assert self.cube.units == "hPa"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32

    def test_per_second(self, mocker):
        with assert_no_warnings_regexp():
            self._call_units(units_str="/s")
        assert self.cube.units == "s^-1"
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32

    def test_unhandled_unit(self, mocker):
        warning_message = "Unhandled units 'kittens' recorded in cube attributes"
        with pytest.warns(match=warning_message):
            self._call_units(units_str="kittens")
        assert self.cube.units == ""
        assert_array_almost_equal(self.cube.data, np.ones_like(self.cube.data))
        assert self.cube.data.dtype == np.float32
        assert self.cube.attributes["invalid_units"] == "kittens"
