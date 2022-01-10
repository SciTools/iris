# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:func:`iris.fileformats.pp_load_rules._convert_time_coords`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import unittest

from cf_units import CALENDAR_360_DAY, CALENDAR_GREGORIAN, Unit
from cftime import datetime as nc_datetime
import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.fileformats.pp import SplittableInt
from iris.fileformats.pp_load_rules import _convert_time_coords
from iris.tests.unit.fileformats import TestField


def _lbtim(ia=0, ib=0, ic=0):
    return SplittableInt(ic + 10 * (ib + 10 * ia), {"ia": 2, "ib": 1, "ic": 0})


def _lbcode(value=None, ix=None, iy=None):
    if value is not None:
        result = SplittableInt(value, {"iy": slice(0, 2), "ix": slice(2, 4)})
    else:
        # N.B. if 'value' is None, both ix and iy must be set.
        result = SplittableInt(
            10000 + 100 * ix + iy, {"iy": slice(0, 2), "ix": slice(2, 4)}
        )
    return result


_EPOCH_HOURS_UNIT = Unit("hours since epoch", calendar=CALENDAR_GREGORIAN)
_HOURS_UNIT = Unit("hours")


class TestLBTIMx0x_SingleTimepoint(TestField):
    def _check_timepoint(self, lbcode, expect_match=True):
        lbtim = _lbtim(ib=0, ic=1)
        t1 = nc_datetime(1970, 1, 1, hour=6, minute=0, second=0)
        t2 = nc_datetime(
            0, 0, 0, calendar=None, has_year_zero=True
        )  # not used in result
        lbft = None  # unused
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
        )
        if expect_match:
            expect_result = [
                (
                    DimCoord(
                        24 * 0.25,
                        standard_name="time",
                        units=_EPOCH_HOURS_UNIT,
                    ),
                    None,
                )
            ]
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
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
        )
        if expect_match:
            expect_result = [
                (
                    DimCoord(
                        24 * 1.125,
                        standard_name="forecast_period",
                        units="hours",
                    ),
                    None,
                ),
                (
                    DimCoord(
                        24 * 9.25,
                        standard_name="time",
                        units=_EPOCH_HOURS_UNIT,
                    ),
                    None,
                ),
                (
                    DimCoord(
                        24 * 8.125,
                        standard_name="forecast_reference_time",
                        units=_EPOCH_HOURS_UNIT,
                    ),
                    None,
                ),
            ]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)

    def test_normal_xy(self):
        self._check_forecast(_lbcode(1))

    def test_non_time_cross_section(self):
        self._check_forecast(_lbcode(ix=1, iy=2))

    def test_time_cross_section(self):
        self._check_forecast(_lbcode(ix=1, iy=20), expect_match=False)

    def test_exact_hours(self):
        lbtim = _lbtim(ib=1, ic=1)
        t1 = nc_datetime(2015, 1, 20, hour=7, minute=0, second=0)
        t2 = nc_datetime(2015, 1, 20, hour=0, minute=0, second=0)
        coords_and_dims = _convert_time_coords(
            lbcode=_lbcode(1),
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=None,
        )
        (fp, _), (t, _), (frt, _) = coords_and_dims
        # These should both be exact whole numbers.
        self.assertEqual(fp.points[0], 7)
        self.assertEqual(t.points[0], 394927)

    def test_not_exact_hours(self):
        lbtim = _lbtim(ib=1, ic=1)
        t1 = nc_datetime(2015, 1, 20, hour=7, minute=10, second=0)
        t2 = nc_datetime(2015, 1, 20, hour=0, minute=0, second=0)
        coords_and_dims = _convert_time_coords(
            lbcode=_lbcode(1),
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=None,
        )
        (fp, _), (t, _), (frt, _) = coords_and_dims
        self.assertArrayAllClose(fp.points[0], 7.1666666, atol=0.0001, rtol=0)
        self.assertArrayAllClose(t.points[0], 394927.166666, atol=0.01, rtol=0)


class TestLBTIMx2x_TimePeriod(TestField):
    def _check_period(self, lbcode, expect_match=True):
        lbtim = _lbtim(ib=2, ic=1)
        # Start time
        t1 = nc_datetime(1970, 1, 9, hour=3, minute=0, second=0)
        # End time
        t2 = nc_datetime(1970, 1, 10, hour=3, minute=0, second=0)
        lbft = 2.0  # sample period
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
        )
        if expect_match:
            expect_result = [
                (
                    DimCoord(
                        24 * 9.125 - 2.0,
                        standard_name="forecast_reference_time",
                        units=_EPOCH_HOURS_UNIT,
                    ),
                    None,
                ),
                (
                    DimCoord(
                        standard_name="forecast_period",
                        units="hours",
                        points=[-10.0],
                        bounds=[-22.0, 2.0],
                    ),
                    None,
                ),
                (
                    DimCoord(
                        standard_name="time",
                        units=_EPOCH_HOURS_UNIT,
                        points=[24 * 8.625],
                        bounds=[24 * 8.125, 24 * 9.125],
                    ),
                    None,
                ),
            ]
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
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
        )
        if expect_match:
            t1_hours = 24 * 8.375
            t2_hours = 24 * (10.375 + 2 * 365)
            period_hours = 24.0 * (2 * 365 + 2)
            expect_result = [
                (
                    DimCoord(
                        [t2_hours - lbft],
                        standard_name="forecast_reference_time",
                        units=_EPOCH_HOURS_UNIT,
                    ),
                    None,
                ),
                (
                    DimCoord(
                        standard_name="forecast_period",
                        units="hours",
                        points=[lbft],
                        bounds=[lbft - period_hours, lbft],
                    ),
                    None,
                ),
                (
                    DimCoord(
                        standard_name="time",
                        units=_EPOCH_HOURS_UNIT,
                        points=[t2_hours],
                        bounds=[t1_hours, t2_hours],
                    ),
                    None,
                ),
            ]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)

    def test_normal_xy(self):
        self._check_yearly(_lbcode(1))

    def test_non_time_cross_section(self):
        self._check_yearly(_lbcode(ix=1, iy=2))

    def test_time_cross_section(self):
        self._check_yearly(_lbcode(ix=1, iy=20), expect_match=False)


class TestLBTIMx2x_ZeroYear(TestField):
    def test_(self):
        lbtim = _lbtim(ib=2, ic=1)
        t1 = nc_datetime(0, 1, 1, has_year_zero=True)
        t2 = nc_datetime(0, 1, 31, 23, 59, 00, has_year_zero=True)
        lbft = 0
        lbcode = _lbcode(1)
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
        )
        self.assertEqual(coords_and_dims, [])


class TestLBTIMxxx_Unhandled(TestField):
    def test_unrecognised(self):
        lbtim = _lbtim(ib=4, ic=1)
        t1 = nc_datetime(0, 0, 0, calendar=None, has_year_zero=True)
        t2 = nc_datetime(0, 0, 0, calendar=None, has_year_zero=True)
        lbft = None
        lbcode = _lbcode(0)
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
        )
        self.assertEqual(coords_and_dims, [])


class TestLBCODE3xx(TestField):
    def test(self):
        lbcode = _lbcode(value=31323)
        lbtim = _lbtim(ib=2, ic=2)
        calendar = CALENDAR_360_DAY
        t1 = nc_datetime(
            1970, 1, 3, hour=0, minute=0, second=0, calendar=calendar
        )
        t2 = nc_datetime(
            1970, 1, 4, hour=0, minute=0, second=0, calendar=calendar
        )
        lbft = 24 * 4
        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
        )
        t2_hours = 24 * 3
        expected_result = [
            (
                DimCoord(
                    [t2_hours - lbft],
                    standard_name="forecast_reference_time",
                    units=_EPOCH_HOURS_UNIT,
                ),
                None,
            )
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected_result)


class TestArrayInputWithLBTIM_0_0_1(TestField):
    def test_t1_list(self):
        # lbtim ia = 0, ib = 0, ic = 1
        # with a series of times (t1).
        lbcode = _lbcode(1)
        lbtim = _lbtim(ia=0, ib=0, ic=1)
        hours = np.array([0, 3, 6, 9, 12])
        # Validity time - vector of different values
        t1 = [nc_datetime(1970, 1, 9, hour=3 + hour) for hour in hours]
        t1_dims = (0,)
        # Forecast reference time - scalar (not used)
        t2 = nc_datetime(1970, 1, 9, hour=3)
        lbft = None

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t1_dims=t1_dims,
        )

        # Expected coords.
        time_coord = DimCoord(
            (24 * 8) + 3 + hours, standard_name="time", units=_EPOCH_HOURS_UNIT
        )
        expected = [(time_coord, (0,))]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)


class TestArrayInputWithLBTIM_0_1_1(TestField):
    def test_t1_list_t2_scalar(self):
        # lbtim ia = 0, ib = 1, ic = 1
        # with a single forecast reference time (t2) and a series
        # of validity times (t1).
        lbcode = _lbcode(1)
        lbtim = _lbtim(ia=0, ib=1, ic=1)
        forecast_period_in_hours = np.array([0, 3, 6, 9, 12])
        # Validity time - vector of different values
        t1 = [
            nc_datetime(1970, 1, 9, hour=(3 + fp))
            for fp in forecast_period_in_hours
        ]
        t1_dims = (0,)
        # Forecast reference time - scalar
        t2 = nc_datetime(1970, 1, 9, hour=3)
        lbft = None  # Not used.

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t1_dims=t1_dims,
        )

        # Expected coords.
        fp_coord = DimCoord(
            forecast_period_in_hours,
            standard_name="forecast_period",
            units="hours",
        )
        time_coord = DimCoord(
            (24 * 8) + 3 + forecast_period_in_hours,
            standard_name="time",
            units=_EPOCH_HOURS_UNIT,
        )
        fref_time_coord = DimCoord(
            (24 * 8) + 3,
            standard_name="forecast_reference_time",
            units=_EPOCH_HOURS_UNIT,
        )
        expected = [
            (fp_coord, (0,)),
            (time_coord, (0,)),
            (fref_time_coord, None),
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)

    def test_t1_and_t2_list(self):
        # lbtim ia = 0, ib = 1, ic = 1
        # with a single repeated forecast reference time (t2) and a series
        # of validity times (t1).
        lbcode = _lbcode(1)
        lbtim = _lbtim(ia=0, ib=1, ic=1)
        forecast_period_in_hours = np.array([0, 3, 6, 9, 12])
        # Validity time - vector of different values
        t1 = [
            nc_datetime(1970, 1, 9, hour=(3 + fp))
            for fp in forecast_period_in_hours
        ]
        t1_dims = (0,)
        # Forecast reference time - vector of same values
        t2 = [
            nc_datetime(1970, 1, 9, hour=3) for _ in forecast_period_in_hours
        ]
        t2_dims = (0,)
        lbft = None  # Not used.

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t1_dims=t1_dims,
            t2_dims=t2_dims,
        )

        # Expected coords.
        fp_coord = DimCoord(
            forecast_period_in_hours,
            standard_name="forecast_period",
            units="hours",
        )
        time_coord = DimCoord(
            (24 * 8) + 3 + forecast_period_in_hours,
            standard_name="time",
            units=_EPOCH_HOURS_UNIT,
        )
        fref_time_coord = DimCoord(
            (24 * 8) + 3,
            standard_name="forecast_reference_time",
            units=_EPOCH_HOURS_UNIT,
        )
        expected = [
            (fp_coord, (0,)),
            (time_coord, (0,)),
            (fref_time_coord, None),
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)

    def test_t1_and_t2_orthogonal_lists(self):
        # lbtim ia = 0, ib = 1, ic = 1
        # with a single repeated forecast reference time (t2) and a series
        # of validity times (t1).
        lbcode = _lbcode(1)
        lbtim = _lbtim(ia=0, ib=1, ic=1)
        years = np.array([1970, 1971, 1972])
        hours = np.array([3, 6, 9, 12])
        # Validity time - vector of different values
        t1 = [nc_datetime(year, 1, 9, hour=12) for year in years]
        t1_dims = (0,)
        # Forecast reference time - vector of different values
        t2 = [nc_datetime(1970, 1, 9, hour=hour) for hour in hours]
        t2_dims = (1,)
        lbft = None  # Not used.

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t1_dims=t1_dims,
            t2_dims=t2_dims,
        )

        # Expected coords.
        points = [
            [(year - 1970) * 365 * 24 + 12 - hour for hour in hours]
            for year in years
        ]
        fp_coord = AuxCoord(
            points, standard_name="forecast_period", units="hours"
        )
        points = (years - 1970) * 24 * 365 + (24 * 8) + 12
        time_coord = DimCoord(
            points, standard_name="time", units=_EPOCH_HOURS_UNIT
        )
        points = (24 * 8) + hours
        fref_time_coord = DimCoord(
            points,
            standard_name="forecast_reference_time",
            units=_EPOCH_HOURS_UNIT,
        )
        expected = [
            (fp_coord, (0, 1)),  # Spans dims 0 and 1.
            (time_coord, (0,)),
            (fref_time_coord, (1,)),
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)

    def test_t1_multi_dim_list_t2_scalar(self):
        # Another case of lbtim ia = 0, ib = 1, ic = 1 but
        # with a changing forecast reference time (t2) and
        # validity time (t1).
        lbcode = _lbcode(1)
        lbtim = _lbtim(ia=0, ib=1, ic=1)
        forecast_period_in_hours = np.array([0, 3, 6, 9, 12])
        years = np.array([1970, 1971, 1972])
        # Validity time - 2d array of different values
        t1 = [
            [
                nc_datetime(year, 1, 9, hour=(3 + fp))
                for fp in forecast_period_in_hours
            ]
            for year in years
        ]
        t1_dims = (0, 1)
        # Forecast reference time - vector of different values
        t2 = nc_datetime(1970, 1, 9, hour=3)
        lbft = None  # Not used.

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t1_dims=t1_dims,
        )

        # Expected coords.
        fp_coord = AuxCoord(
            [
                forecast_period_in_hours + (year - 1970) * 365 * 24
                for year in years
            ],
            standard_name="forecast_period",
            units="hours",
        )
        time_coord = AuxCoord(
            [
                (24 * 8)
                + 3
                + forecast_period_in_hours
                + (year - 1970) * 365 * 24
                for year in years
            ],
            standard_name="time",
            units=_EPOCH_HOURS_UNIT,
        )
        fref_time_coord = DimCoord(
            (24 * 8) + 3,
            standard_name="forecast_reference_time",
            units=_EPOCH_HOURS_UNIT,
        )
        expected = [
            (fp_coord, (0, 1)),
            (time_coord, (0, 1)),
            (fref_time_coord, None),
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)

    def test_t1_and_t2_nparrays(self):
        # lbtim ia = 0, ib = 1, ic = 1
        # with a single repeated forecast reference time (t2) and a series
        # of validity times (t1).
        lbcode = _lbcode(1)
        lbtim = _lbtim(ia=0, ib=1, ic=1)
        forecast_period_in_hours = np.array([0, 3, 6, 9, 12])
        # Validity time - vector of different values
        t1 = np.array(
            [
                nc_datetime(1970, 1, 9, hour=(3 + fp))
                for fp in forecast_period_in_hours
            ]
        )
        t1_dims = (0,)
        # Forecast reference time - vector of same values
        t2 = np.array(
            [nc_datetime(1970, 1, 9, hour=3) for _ in forecast_period_in_hours]
        )
        t2_dims = (0,)
        lbft = None  # Not used.

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t1_dims=t1_dims,
            t2_dims=t2_dims,
        )

        # Expected coords.
        fp_coord = DimCoord(
            forecast_period_in_hours,
            standard_name="forecast_period",
            units="hours",
        )
        time_coord = DimCoord(
            (24 * 8) + 3 + forecast_period_in_hours,
            standard_name="time",
            units=_EPOCH_HOURS_UNIT,
        )
        fref_time_coord = DimCoord(
            (24 * 8) + 3,
            standard_name="forecast_reference_time",
            units=_EPOCH_HOURS_UNIT,
        )
        expected = [
            (fp_coord, (0,)),
            (time_coord, (0,)),
            (fref_time_coord, None),
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)


class TestArrayInputWithLBTIM_0_2_1(TestField):
    def test_t1_list_t2_scalar(self):
        lbtim = _lbtim(ib=2, ic=1)
        lbcode = _lbcode(1)
        hours = np.array([0, 3, 6, 9])
        # Start times - vector
        t1 = [nc_datetime(1970, 1, 9, hour=9 + hour) for hour in hours]
        t1_dims = (0,)
        # End time - scalar
        t2 = nc_datetime(1970, 1, 11, hour=9)
        lbft = 3.0  # Sample period

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t1_dims=t1_dims,
        )

        # Expected coords.
        points = lbft - (48 - hours) / 2.0
        bounds = np.array(
            [lbft - (48 - hours), np.ones_like(hours) * lbft]
        ).transpose()
        fp_coord = AuxCoord(
            points,
            standard_name="forecast_period",
            units="hours",
            bounds=bounds,
        )
        points = 9 * 24 + 9 + (hours / 2.0)
        bounds = np.array(
            [8 * 24 + 9 + hours, np.ones_like(hours) * 10 * 24 + 9]
        ).transpose()
        time_coord = AuxCoord(
            points,
            standard_name="time",
            units=_EPOCH_HOURS_UNIT,
            bounds=bounds,
        )
        points = 10 * 24 + 9 - lbft
        fref_time_coord = DimCoord(
            points,
            standard_name="forecast_reference_time",
            units=_EPOCH_HOURS_UNIT,
        )
        expected = [
            (fp_coord, (0,)),
            (time_coord, (0,)),
            (fref_time_coord, None),
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)


class TestArrayInputWithLBTIM_0_3_1(TestField):
    @unittest.skip("#3508 investigate unit test failure")
    def test_t1_scalar_t2_list(self):
        lbtim = _lbtim(ib=3, ic=1)
        lbcode = _lbcode(1)
        years = np.array([1972, 1973, 1974])
        # Start times - scalar
        t1 = nc_datetime(1970, 1, 9, hour=9)
        # End time - vector
        t2 = [nc_datetime(year, 1, 11, hour=9) for year in years]
        t2_dims = (0,)
        lbft = 3.0  # Sample period

        coords_and_dims = _convert_time_coords(
            lbcode=lbcode,
            lbtim=lbtim,
            epoch_hours_unit=_EPOCH_HOURS_UNIT,
            t1=t1,
            t2=t2,
            lbft=lbft,
            t2_dims=t2_dims,
        )

        # Expected coords.
        points = np.ones_like(years) * lbft
        bounds = np.array(
            [lbft - ((years - 1970) * 365 * 24 + 2 * 24), points]
        ).transpose()
        fp_coord = AuxCoord(
            points,
            standard_name="forecast_period",
            units="hours",
            bounds=bounds,
        )
        points = (years - 1970) * 365 * 24 + 10 * 24 + 9
        bounds = np.array(
            [np.ones_like(points) * (8 * 24 + 9), points]
        ).transpose()
        # The time coordinate is an AuxCoord as the lower bound for each
        # cell is the same so it does not meet the monotonicity requirement.
        time_coord = AuxCoord(
            points,
            standard_name="time",
            units=_EPOCH_HOURS_UNIT,
            bounds=bounds,
        )
        fref_time_coord = DimCoord(
            points - lbft,
            standard_name="forecast_reference_time",
            units=_EPOCH_HOURS_UNIT,
        )
        expected = [
            (fp_coord, (0,)),
            (time_coord, (0,)),
            (fref_time_coord, (0,)),
        ]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expected)


if __name__ == "__main__":
    tests.main()
