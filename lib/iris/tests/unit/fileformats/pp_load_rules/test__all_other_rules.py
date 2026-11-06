# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp_load_rules._all_other_rules` function."""

from unittest import mock

from cf_units import CALENDAR_360_DAY, Unit
from cftime import datetime as nc_datetime
import numpy as np

from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.fileformats.pp import SplittableInt
from iris.fileformats.pp_load_rules import _all_other_rules
from iris.tests.unit.fileformats.pp_load_rules import assert_coords_and_dims_lists_match

# iris.fileformats.pp_load_rules._all_other_rules() returns a tuple of
# of various metadata. This constant is the index into this
# tuple to obtain the cell methods.
CELL_METHODS_INDEX = 5
DIM_COORDS_INDEX = 6
AUX_COORDS_INDEX = 7


class TestCellMethods:
    def test_time_mean(self, mocker):
        # lbproc = 128 -> mean
        # lbtim.ib = 2 -> simple t1 to t2 interval.
        field = mocker.MagicMock(lbproc=128, lbtim=mocker.Mock(ia=0, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("mean", "time")]
        assert res == expected

    def test_hourly_mean(self, mocker):
        # lbtim.ia = 1 -> hourly
        field = mocker.MagicMock(lbproc=128, lbtim=mocker.Mock(ia=1, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("mean", "time", "1 hour")]
        assert res == expected

    def test_daily_mean(self, mocker):
        # lbtim.ia = 24 -> daily
        field = mocker.MagicMock(lbproc=128, lbtim=mocker.Mock(ia=24, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("mean", "time", "24 hour")]
        assert res == expected

    def test_custom_max(self, mocker):
        field = mocker.MagicMock(lbproc=8192, lbtim=mocker.Mock(ia=47, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("maximum", "time", "47 hour")]
        assert res == expected

    def test_daily_min(self, mocker):
        # lbproc = 4096 -> min
        field = mocker.MagicMock(lbproc=4096, lbtim=mocker.Mock(ia=24, ib=2, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("minimum", "time", "24 hour")]
        assert res == expected

    def test_time_mean_over_multiple_years(self, mocker):
        # lbtim.ib = 3 -> interval within a year, over multiple years.
        field = mocker.MagicMock(lbproc=128, lbtim=mocker.Mock(ia=0, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean within years", "time"),
            CellMethod("mean over years", "time"),
        ]
        assert res == expected

    def test_hourly_mean_over_multiple_years(self, mocker):
        field = mocker.MagicMock(lbproc=128, lbtim=mocker.Mock(ia=1, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean within years", "time", "1 hour"),
            CellMethod("mean over years", "time"),
        ]
        assert res == expected

    def test_climatology_max(self, mocker):
        field = mocker.MagicMock(lbproc=8192, lbtim=mocker.Mock(ia=24, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("maximum", "time")]
        assert res == expected

    def test_climatology_min(self, mocker):
        field = mocker.MagicMock(lbproc=4096, lbtim=mocker.Mock(ia=24, ib=3, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("minimum", "time")]
        assert res == expected

    def test_other_lbtim_ib(self, mocker):
        # lbtim.ib = 5 -> non-specific aggregation
        field = mocker.MagicMock(lbproc=4096, lbtim=mocker.Mock(ia=24, ib=5, ic=3))
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [CellMethod("minimum", "time")]
        assert res == expected

    def test_multiple_unordered_lbprocs(self, mocker):
        field = mocker.MagicMock(
            lbproc=192,
            bzx=0,
            bdx=1,
            lbnpt=3,
            lbrow=3,
            lbtim=mocker.Mock(ia=24, ib=5, ic=3),
            lbcode=SplittableInt(1),
            x_bounds=None,
            _x_coord_name=lambda: "longitude",
            _y_coord_name=lambda: "latitude",
            # Not under test but needed for the Mock to play nicely.
            bzy=1,
            bdy=1,
        )
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean", "time"),
            CellMethod("mean", "longitude"),
        ]
        assert res == expected

    def test_multiple_unordered_rotated_lbprocs(self, mocker):
        field = mocker.MagicMock(
            lbproc=192,
            bzx=0,
            bdx=1,
            lbnpt=3,
            lbrow=3,
            lbtim=mocker.Mock(ia=24, ib=5, ic=3),
            lbcode=SplittableInt(101),
            x_bounds=None,
            _x_coord_name=lambda: "grid_longitude",
            _y_coord_name=lambda: "grid_latitude",
            # Not under test but needed for the Mock to play nicely.
            bzy=1,
            bdy=1,
        )
        res = _all_other_rules(field)[CELL_METHODS_INDEX]
        expected = [
            CellMethod("mean", "time"),
            CellMethod("mean", "grid_longitude"),
        ]
        assert res == expected


class TestCrossSectionalTime:
    def test_lbcode3x23(self, mocker):
        time_bounds = np.array(
            [[0.875, 1.125], [1.125, 1.375], [1.375, 1.625], [1.625, 1.875]]
        )
        field = mocker.MagicMock(
            lbproc=0,
            bzx=0,
            bdx=0,
            lbnpt=3,
            lbrow=4,
            t1=nc_datetime(2000, 1, 2, hour=0, minute=0, second=0),
            t2=nc_datetime(2000, 1, 3, hour=0, minute=0, second=0),
            lbtim=mocker.Mock(ia=1, ib=2, ic=2),
            lbcode=SplittableInt(31323, {"iy": slice(0, 2), "ix": slice(2, 4)}),
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
        assert_coords_and_dims_lists_match(res, expected)


class TestLBTIMx2x_ZeroYears:
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
        assert_coords_and_dims_lists_match(res, expected)

    def test_diff_month(self):
        field = self._make_field(lbmon=3, lbmond=4)
        field.mock_add_spec(self._spec)
        res = _all_other_rules(field)[AUX_COORDS_INDEX]

        assert_coords_and_dims_lists_match(res, [])

    def test_nonzero_year(self):
        field = self._make_field(lbyr=1)
        field.mock_add_spec(self._spec)
        res = _all_other_rules(field)[AUX_COORDS_INDEX]

        assert_coords_and_dims_lists_match(res, [])

    def test_nonzero_yeard(self):
        field = self._make_field(lbyrd=1)
        field.mock_add_spec(self._spec)
        res = _all_other_rules(field)[AUX_COORDS_INDEX]

        assert_coords_and_dims_lists_match(res, [])
