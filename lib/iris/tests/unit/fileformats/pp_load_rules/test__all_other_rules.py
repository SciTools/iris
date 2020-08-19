# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `iris.fileformats.pp_load_rules._all_other_rules` function.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

import numpy as np
from cf_units import Unit, CALENDAR_360_DAY
from cftime import datetime as nc_datetime

from iris.fileformats.pp_load_rules import _all_other_rules
from iris.fileformats.pp import SplittableInt
from iris.coords import CellMethod, DimCoord, AuxCoord
from iris.tests.unit.fileformats import TestField


# iris.fileformats.pp_load_rules._all_other_rules() returns a tuple of
# of various metadata. This constant is the index into this
# tuple to obtain the cell methods.
CELL_METHODS_INDEX = 5
DIM_COORDS_INDEX = 6
AUX_COORDS_INDEX = 7


class TestCellMethods(tests.IrisTest):
    def test_time_mean(self):
        # lbproc = 128 -> mean
        # lbtim.ib = 2 -> simple t1 to t2 interval.
        field = mock.MagicMock(lbproc=128, lbtim=mock.Mock(ia=0, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("mean", "time")]
        self.assertEqual(res, expected)

    def test_hourly_mean(self):
        # lbtim.ia = 1 -> hourly
        field = mock.MagicMock(lbproc=128, lbtim=mock.Mock(ia=1, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("mean", "time", "1 hour")]
        self.assertEqual(res, expected)

    def test_daily_mean(self):
        # lbtim.ia = 24 -> daily
        field = mock.MagicMock(lbproc=128, lbtim=mock.Mock(ia=24, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("mean", "time", "24 hour")]
        self.assertEqual(res, expected)

    def test_custom_max(self):
        field = mock.MagicMock(lbproc=8192, lbtim=mock.Mock(ia=47, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("maximum", "time", "47 hour")]
        self.assertEqual(res, expected)

    def test_daily_min(self):
        # lbproc = 4096 -> min
        field = mock.MagicMock(lbproc=4096, lbtim=mock.Mock(ia=24, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("minimum", "time", "24 hour")]
        self.assertEqual(res, expected)

    def test_time_mean_over_multiple_years(self):
        # lbtim.ib = 3 -> interval within a year, over multiple years.
        field = mock.MagicMock(lbproc=128, lbtim=mock.Mock(ia=0, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean within years", "time"),
            CellMethod("mean over years", "time"),
        ]
        self.assertEqual(res, expected)

    def test_hourly_mean_over_multiple_years(self):
        field = mock.MagicMock(lbproc=128, lbtim=mock.Mock(ia=1, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean within years", "time", "1 hour"),
            CellMethod("mean over years", "time"),
        ]
        self.assertEqual(res, expected)

    def test_climatology_max(self):
        field = mock.MagicMock(lbproc=8192, lbtim=mock.Mock(ia=24, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("maximum", "time")]
        self.assertEqual(res, expected)

    def test_climatology_min(self):
        field = mock.MagicMock(lbproc=4096, lbtim=mock.Mock(ia=24, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("minimum", "time")]
        self.assertEqual(res, expected)

    def test_other_lbtim_ib(self):
        # lbtim.ib = 5 -> non-specific aggregation
        field = mock.MagicMock(lbproc=4096, lbtim=mock.Mock(ia=24, ib=5, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("minimum", "time")]
        self.assertEqual(res, expected)

    def test_multiple_unordered_lbprocs(self):
        field = mock.MagicMock(
            lbproc=192,
            bzx=0,
            bdx=1,
            lbnpt=3,
            lbrow=3,
            lbtim=mock.Mock(ia=24, ib=5, ic=3),
            lbcode=SplittableInt(1),
            x_bounds=None,
            _x_coord_name=lambda: "longitude",
            _y_coord_name=lambda: "latitude",
        )
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean", "time"),
            CellMethod("mean", "longitude"),
        ]
        self.assertEqual(res, expected)

    def test_multiple_unordered_rotated_lbprocs(self):
        field = mock.MagicMock(
            lbproc=192,
            bzx=0,
            bdx=1,
            lbnpt=3,
            lbrow=3,
            lbtim=mock.Mock(ia=24, ib=5, ic=3),
            lbcode=SplittableInt(101),
            x_bounds=None,
            _x_coord_name=lambda: "grid_longitude",
            _y_coord_name=lambda: "grid_latitude",
        )
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean", "time"),
            CellMethod("mean", "grid_longitude"),
        ]
        self.assertEqual(res, expected)


class TestCrossSectionalTime(TestField):
    def test_lbcode3x23(self):
        time_bounds = np.array(
            [[0.875, 1.125], [1.125, 1.375], [1.375, 1.625], [1.625, 1.875]]
        )
        field = mock.MagicMock(
            lbproc=0,
            bzx=0,
            bdx=0,
            lbnpt=3,
            lbrow=4,
            t1=nc_datetime(2000, 1, 2, hour=0, minute=0, second=0),
            t2=nc_datetime(2000, 1, 3, hour=0, minute=0, second=0),
            lbtim=mock.Mock(ia=1, ib=2, ic=2),
            lbcode=SplittableInt(
                31323, {"iy": slice(0, 2), "ix": slice(2, 4)}
            ),
            x_bounds=None,
            y_bounds=time_bounds,
            _x_coord_name=lambda: "longitude",
            _y_coord_name=lambda: "latitude",
        )

        spec = [
            "lbtim",
            "lbcode",
            "lbrow",
            "lbnpt",
            "lbproc",
            "lbsrce",
            "lbuser",
            "bzx",
            "bdx",
            "bdy",
            "bmdi",
            "t1",
            "t2",
            "stash",
            "x_bounds",
            "y_bounds",
            "_x_coord_name",
            "_y_coord_name",
        ]
        field.mock_add_spec(spec)
        res = _all_other_rules(field)[DIM_COORDS_INDEX]

        expected_time_points = np.array([1, 1.25, 1.5, 1.75]) + (2000 * 360)
        expected_unit = Unit(
            "days since 0000-01-01 00:00:00", calendar=CALENDAR_360_DAY
        )
        expected = [
            (
                DimCoord(
                    expected_time_points,
                    standard_name="time",
                    units=expected_unit,
                    bounds=time_bounds,
                ),
                0,
            )
        ]
        self.assertCoordsAndDimsListsMatch(res, expected)


class TestLBTIMx2x_ZeroYears(TestField):

    _spec = [
        "lbtim",
        "lbcode",
        "lbrow",
        "lbnpt",
        "lbproc",
        "lbsrce",
        "lbhem",
        "lbuser",
        "bzx",
        "bdx",
        "bdy",
        "bmdi",
        "t1",
        "t2",
        "stash",
        "x_bounds",
        "y_bounds",
        "_x_coord_name",
        "_y_coord_name",
    ]

    def _make_field(
        self,
        lbyr=0,
        lbyrd=0,
        lbmon=3,
        lbmond=3,
        lbft=0,
        bdx=1,
        bdy=1,
        bmdi=0,
        ia=0,
        ib=2,
        ic=1,
        lbcode=SplittableInt(3),
    ):
        return mock.MagicMock(
            lbyr=lbyr,
            lbyrd=lbyrd,
            lbmon=lbmon,
            lbmond=lbmond,
            lbft=lbft,
            bdx=bdx,
            bdy=bdy,
            bmdi=bmdi,
            lbtim=mock.Mock(ia=ia, ib=ib, ic=ic),
            lbcode=lbcode,
        )

    def test_month_coord(self):
        field = self._make_field()
        field.mock_add_spec(self._spec)
        res = _all_other_rules(field)[AUX_COORDS_INDEX]

        expected = [
            (AuxCoord(3, long_name="month_number", units="1"), None),
            (AuxCoord("Mar", long_name="month", units=Unit("no unit")), None),
            (
                DimCoord(
                    points=0,
                    standard_name="forecast_period",
                    units=Unit("hours"),
                ),
                None,
            ),
        ]
        self.assertCoordsAndDimsListsMatch(res, expected)

    def test_diff_month(self):
        field = self._make_field(lbmon=3, lbmond=4)
        field.mock_add_spec(self._spec)
        res = _all_other_rules(field)[AUX_COORDS_INDEX]

        self.assertCoordsAndDimsListsMatch(res, [])

    def test_nonzero_year(self):
        field = self._make_field(lbyr=1)
        field.mock_add_spec(self._spec)
        res = _all_other_rules(field)[AUX_COORDS_INDEX]

        self.assertCoordsAndDimsListsMatch(res, [])

    def test_nonzero_yeard(self):
        field = self._make_field(lbyrd=1)
        field.mock_add_spec(self._spec)
        res = _all_other_rules(field)[AUX_COORDS_INDEX]

        self.assertCoordsAndDimsListsMatch(res, [])


if __name__ == "__main__":
    tests.main()
