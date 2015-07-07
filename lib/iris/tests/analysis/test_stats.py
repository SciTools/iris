# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""
Test the iris.analysis.stats module.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
import iris.analysis.stats as stats


@tests.skip_data
class Test_corr(tests.IrisTest):
    def setUp(self):
        self.cube_a = iris.load_cube(iris.sample_data_path('GloSea4',
                                                           'ensemble_001.pp'))
        self.cube_b = iris.load_cube(iris.sample_data_path('GloSea4',
                                                           'ensemble_002.pp'))

    def test_perfect_corr(self):
        r = stats.pearsonr(self.cube_a, self.cube_a,
                           ['latitude',   'longitude'])
        self.assertArrayEqual(r.data, np.array([1.]*6))

    def test_perfect_corr_all_dims(self):
        r = stats.pearsonr(self.cube_a, self.cube_a)
        self.assertArrayEqual(r.data, np.array([1.]))

    def test_incompatible_cubes(self):
        with self.assertRaises(ValueError):
            stats.pearsonr(self.cube_a, self.cube_b[0, :, :], 'time')

    def test_compatible_cubes(self):
        r = stats.pearsonr(self.cube_a, self.cube_b, ['latitude', 'longitude'])
        self.assertArrayAlmostEqual(r.data, [0.99733591,
                                             0.99501693,
                                             0.99674225,
                                             0.99495268,
                                             0.99217004,
                                             0.99362189])

    def test_4d_cube_2_dims(self):
        real_0_c = iris.coords.AuxCoord(np.int32(0), 'realization')
        real_1_c = iris.coords.AuxCoord(np.int32(1), 'realization')

        # Make cubes merge-able.
        self.cube_a.add_aux_coord(real_0_c)
        self.cube_b.add_aux_coord(real_1_c)
        self.cube_a.remove_coord('forecast_period')
        self.cube_a.remove_coord('forecast_reference_time')
        self.cube_b.remove_coord('forecast_period')
        self.cube_b.remove_coord('forecast_reference_time')
        four_d_cube_a = iris.cube\
            .CubeList([self.cube_a, self.cube_b]).merge()[0]
        self.cube_a.remove_coord('realization')
        self.cube_b.remove_coord('realization')
        self.cube_a.add_aux_coord(real_1_c)
        self.cube_b.add_aux_coord(real_0_c)
        four_d_cube_b = iris.cube\
            .CubeList([self.cube_a, self.cube_b]).merge()[0]

        r = stats.pearsonr(four_d_cube_a, four_d_cube_b,
                           ['latitude', 'longitude'])
        expected_corr = [[0.99733591,
                          0.99501693,
                          0.99674225,
                          0.99495268,
                          0.99217004,
                          0.99362189],
                         [0.99733591,
                          0.99501693,
                          0.99674225,
                          0.99495268,
                          0.99217004,
                          0.99362189]]
        self.assertArrayAlmostEqual(r.data, expected_corr)

    def test_non_existent_coord(self):
        with self.assertRaises(ValueError):
            stats.pearsonr(self.cube_a, self.cube_b, 'bad_coord')


if __name__ == '__main__':
    tests.main()
