# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for :func:`iris.fileformats.pp_rules.convert`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import types

import mock
import numpy as np

from iris.fileformats.pp_rules import convert
from iris.util import guess_coord_axis
from iris.fileformats.pp import SplittableInt
from iris.fileformats.pp import PPField3
import iris.tests.unit.fileformats
import iris.unit


class TestLBVC(iris.tests.unit.fileformats.TestField):
    @staticmethod
    def _is_potm_level_coord(coord):
        return (coord.standard_name == 'air_potential_temperature' and
                coord.attributes['positive'] == 'up')

    @staticmethod
    def _is_model_level_number_coord(coord):
        return (coord.standard_name == 'model_level_number' and
                coord.units.is_dimensionless() and
                coord.attributes['positive'] == 'up')

    @staticmethod
    def _is_level_pressure_coord(coord):
        return (coord.name() == 'level_pressure' and
                coord.units == 'Pa')

    @staticmethod
    def _is_sigma_coord(coord):
        return (coord.name() == 'sigma' and
                coord.units.is_dimensionless())

    @staticmethod
    def _is_soil_model_level_number_coord(coord):
        return (coord.long_name == 'soil_model_level_number' and
                coord.units.is_dimensionless() and
                coord.attributes['positive'] == 'down')

    def test_soil_levels(self):
        level = 1234
        field = mock.MagicMock(lbvc=6, lblev=level)
        self._test_for_coord(field, convert,
                             TestLBVC._is_soil_model_level_number_coord,
                             expected_points=level,
                             expected_bounds=None)

    def test_hybrid_pressure_model_level_number(self):
        level = 5678
        field = mock.MagicMock(lbvc=9, lblev=level,
                               blev=20, brlev=23, bhlev=42,
                               bhrlev=45, brsvd=[17, 40])
        self._test_for_coord(field, convert,
                             TestLBVC._is_model_level_number_coord,
                             expected_points=level,
                             expected_bounds=None)

    def test_hybrid_pressure_delta(self):
        delta_point = 12.0
        delta_lower_bound = 11.0
        delta_upper_bound = 13.0
        field = mock.MagicMock(lbvc=9, lblev=5678,
                               blev=20, brlev=23, bhlev=delta_point,
                               bhrlev=delta_lower_bound,
                               brsvd=[17, delta_upper_bound])
        self._test_for_coord(field, convert,
                             TestLBVC._is_level_pressure_coord,
                             expected_points=delta_point,
                             expected_bounds=[delta_lower_bound,
                                              delta_upper_bound])

    def test_hybrid_pressure_sigma(self):
        sigma_point = 0.5
        sigma_lower_bound = 0.6
        sigma_upper_bound = 0.4
        field = mock.MagicMock(lbvc=9, lblev=5678,
                               blev=sigma_point, brlev=sigma_lower_bound,
                               bhlev=12, bhrlev=11,
                               brsvd=[sigma_upper_bound, 13])
        self._test_for_coord(field, convert, TestLBVC._is_sigma_coord,
                             expected_points=sigma_point,
                             expected_bounds=[sigma_lower_bound,
                                              sigma_upper_bound])

    def test_potential_temperature_levels(self):
        potm_value = 27.32
        field = mock.MagicMock(lbvc=19, blev=potm_value)
        self._test_for_coord(field, convert, TestLBVC._is_potm_level_coord,
                             expected_points=np.array([potm_value]),
                             expected_bounds=None)


class TestLBTIM(iris.tests.unit.fileformats.TestField):
    def test_365_calendar(self):
        f = mock.MagicMock(lbtim=SplittableInt(4, {'ia': 2, 'ib': 1, 'ic': 0}),
                           lbyr=2013, lbmon=1, lbdat=1, lbhr=12, lbmin=0,
                           lbsec=0,
                           spec=PPField3)
        f.time_unit = types.MethodType(PPField3.time_unit, f)
        f.calendar = iris.unit.CALENDAR_365_DAY
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(f)

        def is_t_coord(coord_and_dims):
            coord, dims = coord_and_dims
            return coord.standard_name == 'time'

        coords_and_dims = filter(is_t_coord, aux_coords_and_dims)
        self.assertEqual(len(coords_and_dims), 1)
        coord, dims = coords_and_dims[0]
        self.assertEqual(guess_coord_axis(coord), 'T')
        self.assertEqual(coord.units.calendar, '365_day')

    def base_field(self):
        field = PPField3()
        field.lbfc = 0
        field.bdx = 1
        field.bdy = 1
        field.bmdi = 999
        field.lbproc = 0
        field.lbvc = 0
        field.lbuser = [0] * 7
        field.lbrsvd = [0] * 4
        field.lbsrce = 0
        field.lbcode = 0
        return field

    @staticmethod
    def is_forecast_period(coord):
        return (coord.standard_name == 'forecast_period' and
                coord.units == 'hours')

    @staticmethod
    def is_time(coord):
        return (coord.standard_name == 'time' and
                coord.units == 'hours since epoch')

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

        self._test_for_coord(field, convert, self.is_forecast_period,
                             expected_points=7.5,
                             expected_bounds=[6, 9])

        self._test_for_coord(field, convert, self.is_time,
                             expected_points=24 + 13.5,
                             expected_bounds=[36, 39])

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

        self._test_for_coord(field, convert, self.is_forecast_period,
                             expected_points=lbft,
                             expected_bounds=[36 - 30, lbft])

        self._test_for_coord(field, convert, self.is_time,
                             expected_points=lbft + 30,
                             expected_bounds=[36, lbft + 30])


if __name__ == "__main__":
    tests.main()
