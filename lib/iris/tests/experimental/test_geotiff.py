# (C) British Crown Copyright 2013, Met Office
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

import iris


@iris.tests.skip_data
class TestRasterExport(tests.GraphicsTest):
    def setUp(self):
        import iris.experimental.raster

    def test_load(self):
        cubes = iris.load(tests.get_data_path(('PP', 'uk4', 'uk4par09.pp')),
                          'air_pressure_at_sea_level')
        cube = cubes[0][0]
        # iris.experimental.raster.export(cube, "test.tif")
