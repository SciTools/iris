# (C) British Crown Copyright 2010 - 2016, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy as np

import iris
from iris import load_cube
from iris.analysis._interpolate_private import regrid_to_max_resolution
from iris.cube import Cube
from iris.coords import DimCoord
from iris.coord_systems import GeogCS


@tests.skip_data
class TestRegrid(tests.IrisTest):
    @staticmethod
    def patch_data(cube):
        # Workaround until regrid can handle factories
        for factory in cube.aux_factories:
            cube.remove_aux_factory(factory)

        # Remove coords that share lat/lon dimensions
        dim = cube.coord_dims(cube.coord('grid_longitude'))[0]
        for coord in cube.coords(contains_dimension=dim, dim_coords=False):
            cube.remove_coord(coord)
        dim = cube.coord_dims(cube.coord('grid_latitude'))[0]
        for coord in cube.coords(contains_dimension=dim, dim_coords=False):
            cube.remove_coord(coord)

    def setUp(self):
        self.theta_p_alt_path = tests.get_data_path(
            ('PP', 'COLPEX', 'small_colpex_theta_p_alt.pp'))
        self.theta_constraint = iris.Constraint('air_potential_temperature')
        self.airpress_constraint = iris.Constraint('air_pressure')
        self.level_constraint = iris.Constraint(model_level_number=1)
        self.multi_level_constraint = iris.Constraint(
            model_level_number=lambda c: 1 <= c < 6)
        self.forecast_constraint = iris.Constraint(
            forecast_period=lambda dt: 0.49 < dt < 0.51)

    def test_regrid_max_resolution(self):
        low = Cube(np.arange(12).reshape((3, 4)))
        cs = GeogCS(6371229)
        low.add_dim_coord(DimCoord(np.array([-1, 0, 1], dtype=np.int32), 'latitude', units='degrees', coord_system=cs), 0)
        low.add_dim_coord(DimCoord(np.array([-1, 0, 1, 2], dtype=np.int32), 'longitude', units='degrees', coord_system=cs), 1)

        med = Cube(np.arange(20).reshape((4, 5)))
        cs = GeogCS(6371229)
        med.add_dim_coord(DimCoord(np.array([-1, 0, 1, 2], dtype=np.int32), 'latitude', units='degrees', coord_system=cs), 0)
        med.add_dim_coord(DimCoord(np.array([-2, -1, 0, 1, 2], dtype=np.int32), 'longitude', units='degrees', coord_system=cs), 1)

        high = Cube(np.arange(30).reshape((5, 6)))
        cs = GeogCS(6371229)
        high.add_dim_coord(DimCoord(np.array([-2, -1, 0, 1, 2], dtype=np.int32), 'latitude', units='degrees', coord_system=cs), 0)
        high.add_dim_coord(DimCoord(np.array([-2, -1, 0, 1, 2, 3], dtype=np.int32), 'longitude', units='degrees', coord_system=cs), 1)

        cubes = regrid_to_max_resolution([low, med, high], mode='nearest')
        self.assertCMLApproxData(cubes, ('regrid', 'low_med_high.cml'))


if __name__ == "__main__":
    tests.main()
