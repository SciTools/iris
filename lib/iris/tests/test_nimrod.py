# (C) British Crown Copyright 2010 - 2014, Met Office
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


# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

import iris
import iris.fileformats.nimrod_load_rules as nimrod_load_rules


def mock_nimrod_field():
    field = iris.fileformats.nimrod.NimrodField()
    field.int_mdi = -32767
    field.float32_mdi = -32767.0
    return field


class TestLoad(tests.IrisTest):
    @tests.skip_data
    def test_multi_field_load(self):
        # load a cube with two fields
        cube = iris.load(tests.get_data_path(
            ('NIMROD', 'uk2km', 'WO0000000003452',
             '201007020900_u1096_ng_ey00_visibility0180_screen_2km')))
        self.assertCML(cube, ("nimrod", "load_2flds.cml"))

    def test_orography(self):
        # Mock an orography field we've seen.
        field = mock_nimrod_field()
        cube = iris.cube.Cube(np.arange(100).reshape(10, 10))

        field.dt_year = field.dt_month = field.dt_day = field.int_mdi
        field.dt_hour = field.dt_minute = field.int_mdi
        field.proj_biaxial_ellipsoid = 0
        field.tm_meridian_scaling = 0.999601
        field.field_code = 73
        field.vertical_coord_type = 1
        field.title = "(MOCK) 2km mean orography"
        field.units = "metres"
        field.source = "GLOBE DTM"

        nimrod_load_rules.name(cube, field)
        nimrod_load_rules.units(cube, field)
        nimrod_load_rules.reference_time(cube, field)
        nimrod_load_rules.proj_biaxial_ellipsoid(cube, field)
        nimrod_load_rules.tm_meridian_scaling(cube, field)
        nimrod_load_rules.vertical_coord(cube, field)
        nimrod_load_rules.attributes(cube, field)

        self.assertCML(cube, ("nimrod", "mockography.cml"))

    def test_levels_below_ground(self):
        # Mock a soil temperature field we've seen.
        field = mock_nimrod_field()
        cube = iris.cube.Cube(np.arange(100).reshape(10, 10))

        field.field_code = -1  # Not orography
        field.reference_vertical_coord_type = field.int_mdi  # Not bounded
        field.vertical_coord_type = 12
        field.vertical_coord = 42
        nimrod_load_rules.vertical_coord(cube, field)

        self.assertCML(cube, ("nimrod", "levels_below_ground.cml"))

    def test_period_of_interest(self):
        # mock a pressure field
        field = mock_nimrod_field()
        cube = iris.cube.Cube(np.arange(100).reshape(10, 10))

        field.field_code = 0
        field.vt_year = 2013
        field.vt_month = 5
        field.vt_day = 7
        field.vt_hour = 6
        field.vt_minute = 0
        field.vt_second = 0
        field.dt_year = 2013
        field.dt_month = 5
        field.dt_day = 7
        field.dt_hour = 6
        field.dt_minute = 0
        field.dt_second = 0
        field.period_minutes = 60

        nimrod_load_rules.time(cube, field)

        self.assertCML(cube, ("nimrod", "period_of_interest.cml"))


if __name__ == "__main__":
    tests.main()
