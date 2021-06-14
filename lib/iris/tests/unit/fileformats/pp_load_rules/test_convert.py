# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :func:`iris.fileformats.pp_load_rules.convert`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from types import MethodType
from unittest import mock

import cf_units
import cftime
import numpy as np

from iris.fileformats.pp import STASH, PPField3, SplittableInt
from iris.fileformats.pp_load_rules import convert
import iris.tests.unit.fileformats
from iris.util import guess_coord_axis


def _mock_field(**kwargs):
    # Generate a mock field, but ensure T1 and T2 viable for rules.
    field = mock.MagicMock(
        t1=mock.MagicMock(year=1990, month=3, day=7),
        t2=mock.MagicMock(year=1990, month=3, day=7),
    )
    field.configure_mock(**kwargs)
    return field


class TestLBCODE(iris.tests.unit.fileformats.TestField):
    @staticmethod
    def _is_cross_section_height_coord(coord):
        return (
            coord.standard_name == "height"
            and coord.units == "km"
            and coord.attributes["positive"] == "up"
        )

    def test_cross_section_height_bdy_zero(self):
        lbcode = SplittableInt(19902, {"iy": slice(0, 2), "ix": slice(2, 4)})
        points = np.array([10, 20, 30, 40])
        bounds = np.array([[0, 15], [15, 25], [25, 35], [35, 45]])
        field = _mock_field(lbcode=lbcode, bdy=0, y=points, y_bounds=bounds)
        self._test_for_coord(
            field,
            convert,
            TestLBCODE._is_cross_section_height_coord,
            expected_points=points,
            expected_bounds=bounds,
        )

    def test_cross_section_height_bdy_bmdi(self):
        lbcode = SplittableInt(19902, {"iy": slice(0, 2), "ix": slice(2, 4)})
        points = np.array([10, 20, 30, 40])
        bounds = np.array([[0, 15], [15, 25], [25, 35], [35, 45]])
        bmdi = -1.07374e09
        field = _mock_field(
            lbcode=lbcode, bdy=bmdi, bmdi=bmdi, y=points, y_bounds=bounds
        )
        self._test_for_coord(
            field,
            convert,
            TestLBCODE._is_cross_section_height_coord,
            expected_points=points,
            expected_bounds=bounds,
        )


class TestLBVC(iris.tests.unit.fileformats.TestField):
    @staticmethod
    def _is_potm_level_coord(coord):
        return (
            coord.standard_name == "air_potential_temperature"
            and coord.attributes["positive"] == "up"
        )

    @staticmethod
    def _is_model_level_number_coord(coord):
        return (
            coord.standard_name == "model_level_number"
            and coord.units.is_dimensionless()
            and coord.attributes["positive"] == "up"
        )

    @staticmethod
    def _is_level_pressure_coord(coord):
        return coord.name() == "level_pressure" and coord.units == "Pa"

    @staticmethod
    def _is_sigma_coord(coord):
        return coord.name() == "sigma" and coord.units.is_dimensionless()

    @staticmethod
    def _is_soil_model_level_number_coord(coord):
        return (
            coord.long_name == "soil_model_level_number"
            and coord.units.is_dimensionless()
            and coord.attributes["positive"] == "down"
        )

    @staticmethod
    def _is_soil_depth_coord(coord):
        return (
            coord.standard_name == "depth"
            and coord.units == "m"
            and coord.attributes["positive"] == "down"
        )

    def test_soil_levels(self):
        level = 1234
        field = _mock_field(lbvc=6, lblev=level, brsvd=[0, 0], brlev=0)
        self._test_for_coord(
            field,
            convert,
            self._is_soil_model_level_number_coord,
            expected_points=[level],
            expected_bounds=None,
        )

    def test_soil_depth(self):
        lower, point, upper = 1.2, 3.4, 5.6
        field = _mock_field(lbvc=6, blev=point, brsvd=[lower, 0], brlev=upper)
        self._test_for_coord(
            field,
            convert,
            self._is_soil_depth_coord,
            expected_points=[point],
            expected_bounds=[[lower, upper]],
        )

    def test_hybrid_pressure_model_level_number(self):
        level = 5678
        field = _mock_field(
            lbvc=9,
            lblev=level,
            blev=20,
            brlev=23,
            bhlev=42,
            bhrlev=45,
            brsvd=[17, 40],
        )
        self._test_for_coord(
            field,
            convert,
            TestLBVC._is_model_level_number_coord,
            expected_points=[level],
            expected_bounds=None,
        )

    def test_hybrid_pressure_delta(self):
        delta_point = 12.0
        delta_lower_bound = 11.0
        delta_upper_bound = 13.0
        field = _mock_field(
            lbvc=9,
            lblev=5678,
            blev=20,
            brlev=23,
            bhlev=delta_point,
            bhrlev=delta_lower_bound,
            brsvd=[17, delta_upper_bound],
        )
        self._test_for_coord(
            field,
            convert,
            TestLBVC._is_level_pressure_coord,
            expected_points=[delta_point],
            expected_bounds=[[delta_lower_bound, delta_upper_bound]],
        )

    def test_hybrid_pressure_sigma(self):
        sigma_point = 0.5
        sigma_lower_bound = 0.6
        sigma_upper_bound = 0.4
        field = _mock_field(
            lbvc=9,
            lblev=5678,
            blev=sigma_point,
            brlev=sigma_lower_bound,
            bhlev=12,
            bhrlev=11,
            brsvd=[sigma_upper_bound, 13],
        )
        self._test_for_coord(
            field,
            convert,
            TestLBVC._is_sigma_coord,
            expected_points=[sigma_point],
            expected_bounds=[[sigma_lower_bound, sigma_upper_bound]],
        )

    def test_potential_temperature_levels(self):
        potm_value = 27.32
        field = _mock_field(lbvc=19, blev=potm_value)
        self._test_for_coord(
            field,
            convert,
            TestLBVC._is_potm_level_coord,
            expected_points=np.array([potm_value]),
            expected_bounds=None,
        )


class TestLBTIM(iris.tests.unit.fileformats.TestField):
    def test_365_calendar(self):
        f = mock.MagicMock(
            lbtim=SplittableInt(4, {"ia": 2, "ib": 1, "ic": 0}),
            lbyr=2013,
            lbmon=1,
            lbdat=1,
            lbhr=12,
            lbmin=0,
            lbsec=0,
            t1=cftime.datetime(2013, 1, 1, 12, 0, 0),
            t2=cftime.datetime(2013, 1, 2, 12, 0, 0),
            spec=PPField3,
        )
        f.time_unit = MethodType(PPField3.time_unit, f)
        f.calendar = cf_units.CALENDAR_365_DAY
        (
            factories,
            references,
            standard_name,
            long_name,
            units,
            attributes,
            cell_methods,
            dim_coords_and_dims,
            aux_coords_and_dims,
        ) = convert(f)

        def is_t_coord(coord_and_dims):
            coord, dims = coord_and_dims
            return coord.standard_name == "time"

        coords_and_dims = list(filter(is_t_coord, aux_coords_and_dims))
        self.assertEqual(len(coords_and_dims), 1)
        coord, dims = coords_and_dims[0]
        self.assertEqual(guess_coord_axis(coord), "T")
        self.assertEqual(coord.units.calendar, "365_day")

    def base_field(self):
        field = PPField3(header=mock.MagicMock())
        field.lbfc = 0
        field.bdx = 1
        field.bdy = 1
        field.bmdi = 999
        field.lbproc = 0
        field.lbvc = 0
        field.lbuser = [0] * 7
        field.lbrsvd = [0] * 4
        field.brsvd = [0] * 4
        field.lbsrce = 0
        field.lbcode = 0
        return field

    @staticmethod
    def is_forecast_period(coord):
        return (
            coord.standard_name == "forecast_period" and coord.units == "hours"
        )

    @staticmethod
    def is_time(coord):
        return (
            coord.standard_name == "time"
            and coord.units == "hours since epoch"
        )

    def test_time_mean_ib2(self):
        field = self.base_field()
        field.lbtim = 21
        # Implicit reference time: 1970-01-02 06:00
        field.lbft = 9
        # t1
        field.lbyr, field.lbmon, field.lbdat = 1970, 1, 2
        field.lbhr, field.lbmin, field.lbsec = 12, 0, 0
        # t2
        field.lbyrd, field.lbmond, field.lbdatd = 1970, 1, 2
        field.lbhrd, field.lbmind, field.lbsecd = 15, 0, 0

        self._test_for_coord(
            field,
            convert,
            self.is_forecast_period,
            expected_points=[7.5],
            expected_bounds=[[6, 9]],
        )

        self._test_for_coord(
            field,
            convert,
            self.is_time,
            expected_points=[24 + 13.5],
            expected_bounds=[[36, 39]],
        )

    def test_time_mean_ib3(self):
        field = self.base_field()
        field.lbtim = 31
        # Implicit reference time: 1970-01-02 06:00
        field.lbft = lbft = ((365 + 1) * 24 + 15) - (24 + 6)
        # t1
        field.lbyr, field.lbmon, field.lbdat = 1970, 1, 2
        field.lbhr, field.lbmin, field.lbsec = 12, 0, 0
        # t2
        field.lbyrd, field.lbmond, field.lbdatd = 1971, 1, 2
        field.lbhrd, field.lbmind, field.lbsecd = 15, 0, 0

        self._test_for_coord(
            field,
            convert,
            self.is_forecast_period,
            expected_points=[lbft],
            expected_bounds=[[36 - 30, lbft]],
        )

        self._test_for_coord(
            field,
            convert,
            self.is_time,
            expected_points=[lbft + 30],
            expected_bounds=[[36, lbft + 30]],
        )


class TestLBRSVD(iris.tests.unit.fileformats.TestField):
    @staticmethod
    def _is_realization(coord):
        return coord.standard_name == "realization" and coord.units == "1"

    def test_realization(self):
        lbrsvd = [0] * 4
        lbrsvd[3] = 71
        points = np.array([71])
        bounds = None
        field = _mock_field(lbrsvd=lbrsvd)
        self._test_for_coord(
            field,
            convert,
            TestLBRSVD._is_realization,
            expected_points=points,
            expected_bounds=bounds,
        )


class TestLBSRCE(iris.tests.IrisTest):
    def check_um_source_attrs(
        self, lbsrce, source_str=None, um_version_str=None
    ):
        field = _mock_field(lbsrce=lbsrce)
        (
            factories,
            references,
            standard_name,
            long_name,
            units,
            attributes,
            cell_methods,
            dim_coords_and_dims,
            aux_coords_and_dims,
        ) = convert(field)
        if source_str is not None:
            self.assertEqual(attributes["source"], source_str)
        else:
            self.assertNotIn("source", attributes)
        if um_version_str is not None:
            self.assertEqual(attributes["um_version"], um_version_str)
        else:
            self.assertNotIn("um_version", attributes)

    def test_none(self):
        self.check_um_source_attrs(
            lbsrce=8123, source_str=None, um_version_str=None
        )

    def test_no_um_version(self):
        self.check_um_source_attrs(
            lbsrce=1111,
            source_str="Data from Met Office Unified Model",
            um_version_str=None,
        )

    def test_um_version(self):
        self.check_um_source_attrs(
            lbsrce=12071111,
            source_str="Data from Met Office Unified Model",
            um_version_str="12.7",
        )


class Test_STASH_CF(iris.tests.unit.fileformats.TestField):
    def test_stash_cf_air_temp(self):
        lbuser = [1, 0, 0, 16203, 0, 0, 1]
        lbfc = 16
        stash = STASH(lbuser[6], lbuser[3] // 1000, lbuser[3] % 1000)
        field = _mock_field(lbuser=lbuser, lbfc=lbfc, stash=stash)
        (
            factories,
            references,
            standard_name,
            long_name,
            units,
            attributes,
            cell_methods,
            dim_coords_and_dims,
            aux_coords_and_dims,
        ) = convert(field)
        self.assertEqual(standard_name, "air_temperature")
        self.assertEqual(units, "K")

    def test_no_std_name(self):
        lbuser = [1, 0, 0, 0, 0, 0, 0]
        lbfc = 0
        stash = STASH(lbuser[6], lbuser[3] // 1000, lbuser[3] % 1000)
        field = _mock_field(lbuser=lbuser, lbfc=lbfc, stash=stash)
        (
            factories,
            references,
            standard_name,
            long_name,
            units,
            attributes,
            cell_methods,
            dim_coords_and_dims,
            aux_coords_and_dims,
        ) = convert(field)
        self.assertIsNone(standard_name)
        self.assertIsNone(units)


class Test_LBFC_CF(iris.tests.unit.fileformats.TestField):
    def test_fc_cf_air_temp(self):
        lbuser = [1, 0, 0, 0, 0, 0, 0]
        lbfc = 16
        stash = STASH(lbuser[6], lbuser[3] // 1000, lbuser[3] % 1000)
        field = _mock_field(lbuser=lbuser, lbfc=lbfc, stash=stash)
        (
            factories,
            references,
            standard_name,
            long_name,
            units,
            attributes,
            cell_methods,
            dim_coords_and_dims,
            aux_coords_and_dims,
        ) = convert(field)
        self.assertEqual(standard_name, "air_temperature")
        self.assertEqual(units, "K")


if __name__ == "__main__":
    tests.main()
