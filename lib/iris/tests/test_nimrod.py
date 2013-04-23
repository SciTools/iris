# (C) British Crown Copyright 2010 - 2012, Met Office
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

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

import iris
import iris.quickplot as qplt
import iris.fileformats.nimrod_load_rules as nimrod_load_rules


def mock_nimrod_field():
    field = iris.fileformats.nimrod.NimrodField()
    field.int_mdi = -32767
    field.float32_mdi = -32767.0
    return field


@iris.tests.skip_data
class TestLoad(tests.GraphicsTest):

    def test_load(self):
        cube = iris.load(
            tests.get_data_path(
                ('NIMROD', 'uk2km', 'WO0000000003452',
                 '201007020900_u1096_ng_ey00_visibility0180_screen_2km')))[0]
        self.assertCML(cube, ("nimrod", "load.cml"))

        ax = plt.subplot(1, 1, 1, projection=ccrs.OSGB())
        qplt.contourf(cube, coords=["projection_x_coordinate",
                                    "projection_y_coordinate"],
                      levels=np.linspace(-25000, 6000, 10))
        ax.coastlines()
        self.check_graphic()

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


if __name__ == "__main__":
    tests.main()
