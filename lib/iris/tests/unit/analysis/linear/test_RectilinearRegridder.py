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
"""Unit tests for :class:`iris.analysis._linear.RectilinearRegridder`."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.analysis._linear import RectilinearRegridder
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.cube import Cube


class Test(tests.IrisTest):
    def cube(self, x, y):
        data = np.arange(len(x) * len(y)).reshape(len(y), len(x))
        cube = Cube(data)
        lat = DimCoord(y, 'latitude', units='degrees')
        lon = DimCoord(x, 'longitude', units='degrees')
        cube.add_dim_coord(lat, 0)
        cube.add_dim_coord(lon, 1)
        return cube

    def grids(self):
        src = self.cube(np.linspace(20, 30, 3), np.linspace(10, 25, 4))
        target = self.cube(np.linspace(6, 18, 8), np.linspace(11, 22, 9))
        return src, target

    def extract_grid(self, cube):
        return cube.coord('latitude'), cube.coord('longitude')

    def check_mode(self, mode):
        src_grid, target_grid = self.grids()
        kwargs = {'extrapolation_mode': mode}
        regridder = RectilinearRegridder(src_grid, target_grid, **kwargs)
        # Make a new cube to regrid with different data so we can
        # distinguish between regridding the original src grid
        # definition cube and the cube passed to the regridder.
        src = src_grid.copy()
        src.data += 10

        # To avoid duplicating tests, just check the RectilinearRegridder
        # defers to the experimental regrid function with the correct
        # arguments (and honouring the return value!)
        with mock.patch('iris.experimental.regrid.'
                        'regrid_bilinear_rectilinear_src_and_grid',
                        return_value=mock.sentinel.result) as regrid:
            result = regridder(src)
        self.assertEqual(regrid.call_count, 1)
        _, args, kwargs = regrid.mock_calls[0]
        self.assertEqual(args[0], src)
        self.assertEqual(self.extract_grid(args[1]),
                         self.extract_grid(target_grid))
        self.assertEqual(kwargs, {'extrapolation_mode': mode})
        self.assertIs(result, mock.sentinel.result)

    def test_mode_error(self):
        self.check_mode('error')

    def test_mode_extrapolate(self):
        self.check_mode('extrapolate')

    def test_mode_nan(self):
        self.check_mode('nan')

    def test_mode_mask(self):
        self.check_mode('mask')

    def test_mode_nanmask(self):
        self.check_mode('nanmask')

    def test_invalid_mode(self):
        src, target = self.grids()
        with self.assertRaises(ValueError):
            RectilinearRegridder(src, target, 'magic')

    def test_mismatched_src_coord_systems(self):
        src = Cube(np.zeros((3, 4)))
        cs = GeogCS(6543210)
        lat = DimCoord(range(3), 'latitude', coord_system=cs)
        lon = DimCoord(range(4), 'longitude')
        src.add_dim_coord(lat, 0)
        src.add_dim_coord(lon, 1)
        target = mock.Mock()
        with self.assertRaises(ValueError):
            RectilinearRegridder(src, target, 'extrapolate')


if __name__ == '__main__':
    tests.main()
