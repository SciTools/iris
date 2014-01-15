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

import mock
import types

from iris.fileformats.pp_rules import convert
from iris.util import guess_coord_axis
from iris.fileformats.pp import SplittableInt
from iris.fileformats.pp import PPField3
import iris.unit


class TestLBVC(tests.IrisTest):
    def test_soil_levels(self):
        field = mock.MagicMock(lbvc=6, blev=1234)
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(field)

        def is_model_level_coord(coord_and_dims):
            coord, dims = coord_and_dims
            return coord.standard_name == 'model_level_number'

        coords_and_dims = filter(is_model_level_coord, aux_coords_and_dims)
        self.assertEqual(len(coords_and_dims), 1)
        coord, dims = coords_and_dims[0]
        self.assertEqual(coord.points, 1234)
        self.assertIsNone(coord.bounds)
        self.assertEqual(guess_coord_axis(coord), 'Z')

    def test_potential_temperature_levels(self):
        potm_value = 27.32
        field = mock.MagicMock(lbvc=19, blev=potm_value)
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(field)

        def is_potm_level_coord(coord_and_dims):
            coord, dims = coord_and_dims
            return coord.standard_name == 'air_potential_temperature'

        coords_and_dims = filter(is_potm_level_coord, aux_coords_and_dims)
        self.assertEqual(len(coords_and_dims), 1)
        coord, dims = coords_and_dims[0]
        self.assertArrayEqual(coord.points, [potm_value])
        self.assertIsNone(coord.bounds)
        self.assertEqual(guess_coord_axis(coord), 'Z')


class TestLBTIM(tests.IrisTest):
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

if __name__ == "__main__":
    tests.main()
