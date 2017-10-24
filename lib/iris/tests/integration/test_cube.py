# (C) British Crown Copyright 2014 - 2017, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris.analysis import MEAN
from iris.cube import Cube

from iris._lazy_data import is_lazy_data, as_lazy_data


class Test_aggregated_by(tests.IrisTest):
    @tests.skip_data
    def test_agg_by_aux_coord(self):
        problem_test_file = tests.get_data_path(('NetCDF', 'testing',
                                                'small_theta_colpex.nc'))
        cube = iris.load_cube(problem_test_file, 'air_potential_temperature')

        # Test aggregating by aux coord, notably the `forecast_period` aux
        # coord on `cube`, whose `_points` attribute is a lazy array.
        # This test then ensures that aggregating using `points` instead is
        # successful.

        # First confirm we've got a lazy array.
        # NB. This checks the merge process in `load_cube()` hasn't
        # triggered the load of the coordinate's data.
        forecast_period_coord = cube.coord('forecast_period')

        self.assertTrue(is_lazy_data(forecast_period_coord.core_points()))

        # Now confirm we can aggregate along this coord.
        res_cube = cube.aggregated_by('forecast_period', MEAN)
        res_cell_methods = res_cube.cell_methods[0]
        self.assertEqual(res_cell_methods.coord_names, ('forecast_period',))
        self.assertEqual(res_cell_methods.method, 'mean')


class TestDataFillValue(tests.IrisTest):
    def test_real(self):
        data = np.ma.masked_array([1, 2, 3], [0, 1, 0], fill_value=10)
        cube = Cube(data)
        cube.data.fill_value = 20
        self.assertEqual(cube.data.fill_value, 20)

    def test_lazy(self):
        data = np.ma.masked_array([1, 2, 3], [0, 1, 0], fill_value=10)
        data = as_lazy_data(data)
        cube = Cube(data)
        cube.data.fill_value = 20
        self.assertEqual(cube.data.fill_value, 20)


if __name__ == '__main__':
    tests.main()
