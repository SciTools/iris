# (C) British Crown Copyright 2014, Met Office
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
"""Integration tests for :class:`iris.cube.Cube`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris
from iris.analysis import MEAN
from iris.coords import AuxCoord, CellMethod


class Test_aggregated_by(tests.IrisTest):
    def setUp(self):
        problem_test_file = tests.get_data_path(('NetCDF', 'testing',
                                                'small_theta_colpex.nc'))
        self.cube = iris.load_cube(problem_test_file)

    def test_agg_by_aux_coord(self):
        # Test aggregating by aux coord, notably the `forecast_period` aux
        # coord on `self.cube`, whose `_points` attribute is of type
        # :class:`iris.aux_coords.LazyArray`. This test then ensures that
        # aggregating using `points` instead is successful.

        res_cube = self.cube.aggregated_by('forecast_period', MEAN)
        res_cell_methods = res_cube.cell_methods[0]
        self.assertEqual(res_cell_methods.coord_names, ('forecast_period',))
        self.assertEqual(res_cell_methods.method, 'mean')


if __name__ == '__main__':
    tests.main()
