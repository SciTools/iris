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
:func:`iris.fileformats.pp_rules._convert_time_coords`.

NOTE: basic calculations logic is tested in "test__convert_scalar_time_coords".
Here we are testing the vector/array coordinate specifics.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
from netcdftime import datetime as nc_datetime
import numpy as np

import iris
from iris.coords import DimCoord, AuxCoord
from iris.fileformats.pp import SplittableInt
from iris.fileformats.pp_rules import _convert_time_coords
from iris.tests.unit.fileformats import TestField


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


_EPOCH_HOURS_UNIT = iris.unit.Unit('hours since epoch',
                                   calendar=iris.unit.CALENDAR_GREGORIAN)

_HOURS_UNIT = iris.unit.Unit('hours')


class TestLBTIMx0x_SingleTimepoint(TestField):
    def _check_timepoint(self, lbcode, expect_match=True):
        lbtim = _lbtim(ib=0, ic=1)
        t1 = nc_datetime(1970, 1, 1, hour=6, minute=0, second=0)
        t2 = nc_datetime(0, 0, 0)  # not used in result
        lbft = None  # unused
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            expect_result = [
                (DimCoord(24 * 0.25, standard_name='time',
                          units=_EPOCH_HOURS_UNIT),
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
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            expect_result = [
                (DimCoord(24 * 1.125,
                          standard_name='forecast_period', units='hours'),
                 None),
                (DimCoord(24 * 9.25,
                          standard_name='time', units=_EPOCH_HOURS_UNIT),
                 None),
                (DimCoord(24 * 8.125,
                          standard_name='forecast_reference_time',
                          units=_EPOCH_HOURS_UNIT),
                 None)]
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
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            expect_result = [
                (DimCoord(24 * 9.125 - 2.0,
                          standard_name='forecast_reference_time',
                          units=_EPOCH_HOURS_UNIT), None),
                (DimCoord(standard_name='forecast_period', units='hours',
                          points=[-10.0], bounds=[-22.0, 2.0]), None),
                (DimCoord(standard_name='time', units=_EPOCH_HOURS_UNIT,
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
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        if expect_match:
            t1_hours = 24 * 8.375
            t2_hours = 24 * (10.375 + 2 * 365)
            period_hours = 24.0 * (2 * 365 + 2)
            expect_result = [
                (DimCoord([t2_hours - lbft],
                          standard_name='forecast_reference_time',
                          units=_EPOCH_HOURS_UNIT), None),
                (DimCoord(standard_name='forecast_period', units='hours',
                          points=[lbft], bounds=[lbft - period_hours, lbft]),
                 None),
                (DimCoord(standard_name='time', units=_EPOCH_HOURS_UNIT,
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
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode, lbtim=lbtim, epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1, t2=t2, lbft=lbft)
        self.assertEqual(coords_and_dims, [])


class Test__vector_calls(TestField):
    def test_reshape_call(self):
        # Check that "_reshape_vector_args" is called correctly.
        odd_mocks = mock.MagicMock()
        reshape_call = mock.MagicMock(
            return_value=(odd_mocks.a_t1, odd_mocks.a_t2, odd_mocks.a_ft))
        with mock.patch('iris.fileformats.pp_rules._reshape_vector_args',
                        new=reshape_call):
            coords_and_dims = _convert_time_coords(
                lbcode=odd_mocks.lbcode,
                lbtim=odd_mocks.lbtim,
                epoch_hours_unit=odd_mocks.epoch_unit,
                t1=odd_mocks.t1, t1_dims=odd_mocks.t1_dims,
                t2=odd_mocks.t2, t2_dims=odd_mocks.t2_dims,
                lbft=odd_mocks.lbft, lbft_dims=odd_mocks.lbft_dims)
        # Check it was called once with the expected args.
        self.assertEqual(reshape_call.call_count, 1)
        self.assertEqual(reshape_call.call_args,
                         mock.call([(odd_mocks.t1, odd_mocks.t1_dims),
                                    (odd_mocks.t2, odd_mocks.t2_dims),
                                    (odd_mocks.lbft, odd_mocks.lbft_dims)]))

    def _check_reduce_calls(self, lbtim_ib, n_coords_expected):
        # Check that "_reduce_points_and_bounds" is called correctly, for a
        # given setting of LBTIM.IB.
        lbtim = _lbtim(ib=lbtim_ib, ic=1)
        # Provide dummy times and units that won't crash the testee.
        times = _EPOCH_HOURS_UNIT.num2date(np.arange(2))
        reshape_call = mock.MagicMock(
            return_value=(times, times, np.arange(2)))
        # Make distinguishable test data for 1st, 2nd, 3rd coords returned.
        results_array = [((0,), [1, 2, 3], None),
                         ((0,), [3, 4, 5], None),
                         ((0,), [5, 6, 7], None)]
        reduce_call = mock.MagicMock(side_effect=results_array)
        odd_mocks = mock.MagicMock()
        with \
            mock.patch('iris.fileformats.pp_rules._reshape_vector_args',
                       reshape_call), \
            mock.patch('iris.fileformats.pp_rules._reduce_points_and_bounds',
                       reduce_call):
                coords_and_dims = _convert_time_coords(
                    lbcode=mock.MagicMock(),
                    lbtim=lbtim,
                    epoch_hours_unit=_EPOCH_HOURS_UNIT,
                    t1=odd_mocks.t1, t1_dims=odd_mocks.t1_dims,
                    t2=odd_mocks.t2, t2_dims=odd_mocks.t2_dims,
                    lbft=odd_mocks.lbft, lbft_dims=odd_mocks.lbft_dims)
        # Check reduce was called the correct number of times.
        self.assertEqual(reduce_call.call_count, n_coords_expected)
        # Check the number of coords_and_dims is the same.
        self.assertEqual(len(coords_and_dims), n_coords_expected)
        # Check we got the expected coords in the expected order.
        if n_coords_expected == 1:
            names = ['time']
        else:
            names = ['forecast_period', 'time', 'forecast_reference_time']
        for i_result, (coord, dims) in enumerate(coords_and_dims):
            result = results_array[i_result]
            self.assertEqual(coord.name(), names[i_result])
            self.assertArrayEqual(coord.points, result[1])

    def test_reduce_LBTIMx0x(self):
        self._check_reduce_calls(lbtim_ib=0, n_coords_expected=1)

    def test_reduce_LBTIMx1x(self):
        self._check_reduce_calls(lbtim_ib=1, n_coords_expected=3)

    def test_reduce_LBTIMx2x(self):
        self._check_reduce_calls(lbtim_ib=2, n_coords_expected=3)

    def test_reduce_LBTIMx3x(self):
        self._check_reduce_calls(lbtim_ib=3, n_coords_expected=3)


if __name__ == "__main__":
    tests.main()
