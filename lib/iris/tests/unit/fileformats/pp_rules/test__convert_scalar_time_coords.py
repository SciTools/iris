# (C) British Crown Copyright 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Unit tests for
:func:`iris.fileformats.pp_rules._convert_scalar_time_coords`.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from netcdftime import datetime as nc_datetime

from iris.coords import DimCoord
from iris.fileformats.pp import SplittableInt
from iris.fileformats.pp_rules import _convert_scalar_time_coords
from iris.tests.unit.fileformats import TestField
from iris.unit import Unit, CALENDAR_GREGORIAN


def _lbtim(ia=0, ib=0, ic=0):
    return SplittableInt(ic + 10 * (ib + 10 * ia), {'ia': 2, 'ib': 1, 'ic': 0})


def _lbcode(value=None, ix=None, iy=None):
    if value is not None:
        result = SplittableInt(value, {'iy': slice(0, 2), 'ix': slice(2, 4)})
    else:
        # N.B. if 'value' is None, both ix and iy must be set.
        result = SplittableInt(10000 + 100 * ix + iy,
                               {'iy': slice(0, 2), 'ix': slice(2, 4)})
    return result


_EPOCH_TIME_UNIT = Unit('hours since epoch', calendar=CALENDAR_GREGORIAN)


class TestLBTIMx0x_SingleTimepoint(TestField):
    def _check_timepoint(self, lbcode, expect_match=True):
        lbtim = _lbtim(ib=0, ic=1)
        t1 = nc_datetime(1970, 1, 1, hour=6, minute=0, second=0)
        t2 = nc_datetime(0, 0, 0)  # not used in result
        lbft = None  # unused
        coords_and_dims = _convert_scalar_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_TIME_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            expect_result = [
                (DimCoord(24 * 0.25, standard_name='time',
                          units=_EPOCH_TIME_UNIT),
                 None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)

    def test_normal_xy_dims(self):
        self._check_timepoint(_lbcode(1))

    def test_non_time_cross_section(self):
        self._check_timepoint(_lbcode(ix=1, iy=2))

    def test_time_cross_section(self):
        self._check_timepoint(_lbcode(ix=1, iy=20), expect_match=False)


class TestLBTIMx1x_Forecast(TestField):
    def _check_forecast(self, lbcode, expect_match=True):
        lbtim = _lbtim(ib=1, ic=1)
        # Validity time
        t1 = nc_datetime(1970, 1, 10, hour=6, minute=0, second=0)
        # Forecast time
        t2 = nc_datetime(1970, 1, 9, hour=3, minute=0, second=0)
        lbft = None  # unused
        coords_and_dims = _convert_scalar_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_TIME_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            expect_result = [
                (DimCoord(24 * 1.125,
                          standard_name='forecast_period', units='hours'),
                 None),
                (DimCoord(24 * 9.25,
                          standard_name='time', units=_EPOCH_TIME_UNIT), None),
                (DimCoord(24 * 8.125,
                          standard_name='forecast_reference_time',
                          units=_EPOCH_TIME_UNIT), None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)

    def test_normal_xy(self):
        self._check_forecast(_lbcode(1))

    def test_non_time_cross_section(self):
        self._check_forecast(_lbcode(ix=1, iy=2))

    def test_time_cross_section(self):
        self._check_forecast(_lbcode(ix=1, iy=20), expect_match=False)


class TestLBTIMx2x_TimePeriod(TestField):
    def _check_period(self, lbcode, expect_match=True):
        lbtim = _lbtim(ib=2, ic=1)
        # Start time
        t1 = nc_datetime(1970, 1, 9, hour=3, minute=0, second=0)
        # End time
        t2 = nc_datetime(1970, 1, 10, hour=3, minute=0, second=0)
        lbft = 2.0  # sample period
        coords_and_dims = _convert_scalar_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_TIME_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            expect_result = [
                (DimCoord(24 * 9.125 - 2.0,
                          standard_name='forecast_reference_time',
                          units=_EPOCH_TIME_UNIT), None),
                (DimCoord(standard_name='forecast_period', units='hours',
                          points=[-10.0], bounds=[-22.0, 2.0]), None),
                (DimCoord(standard_name='time', units=_EPOCH_TIME_UNIT,
                          points=[24 * 8.625],
                          bounds=[24 * 8.125, 24 * 9.125]), None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)

    def test_normal_xy(self):
        self._check_period(_lbcode(1))

    def test_non_time_cross_section(self):
        self._check_period(_lbcode(ix=1, iy=2))

    def test_time_cross_section(self):
        self._check_period(_lbcode(ix=1, iy=20), expect_match=False)


class TestLBTIMx3x_YearlyAggregation(TestField):
    def _check_yearly(self, lbcode, expect_match=True):
        lbtim = _lbtim(ib=3, ic=1)
        # Start time
        t1 = nc_datetime(1970, 1, 9, hour=9, minute=0, second=0)
        # End time
        t2 = nc_datetime(1972, 1, 11, hour=9, minute=0, second=0)
        lbft = 3.0  # sample period
        coords_and_dims = _convert_scalar_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_TIME_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            t1_hours = 24 * 8.375
            t2_hours = 24 * (10.375 + 2 * 365)
            period_hours = 24.0 * (2 * 365 + 2)
            expect_result = [
                (DimCoord([t2_hours - lbft],
                          standard_name='forecast_reference_time',
                          units=_EPOCH_TIME_UNIT), None),
                (DimCoord(standard_name='forecast_period', units='hours',
                          points=[lbft], bounds=[lbft - period_hours, lbft]),
                 None),
                (DimCoord(standard_name='time', units=_EPOCH_TIME_UNIT,
                          points=[t2_hours],
                          bounds=[t1_hours, t2_hours]), None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)

    def test_normal_xy(self):
        self._check_yearly(_lbcode(1))

    def test_non_time_cross_section(self):
        self._check_yearly(_lbcode(ix=1, iy=2))

    def test_time_cross_section(self):
        self._check_yearly(_lbcode(ix=1, iy=20), expect_match=False)


class TestLBTIMxxx_Unhandled(TestField):
    def test_unrecognised(self):
        lbtim = _lbtim(ib=4, ic=1)
        t1 = nc_datetime(0, 0, 0)
        t2 = nc_datetime(0, 0, 0)
        lbft = None
        lbcode = _lbcode(0)
        coords_and_dims = _convert_scalar_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_TIME_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        self.assertEqual(coords_and_dims, [])


if __name__ == "__main__":
    tests.main()
