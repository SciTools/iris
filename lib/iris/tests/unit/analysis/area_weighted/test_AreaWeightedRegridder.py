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
Unit tests for :class:`iris.analysis._area_weighted.AreaWeightedRegridder`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.analysis._area_weighted import AreaWeightedRegridder
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

    def check_mdtol(self, mdtol=None):
        src_grid, target_grid = self.grids()
        if mdtol is None:
            regridder = AreaWeightedRegridder(src_grid, target_grid)
            mdtol = 1
        else:
            regridder = AreaWeightedRegridder(src_grid, target_grid,
                                              mdtol=mdtol)

        # Make a new cube to regrid with different data so we can
        # distinguish between regridding the original src grid
        # definition cube and the cube passed to the regridder.
        src = src_grid.copy()
        src.data += 10

        with mock.patch('iris.experimental.regrid.'
                        'regrid_area_weighted_rectilinear_src_and_grid',
                        return_value=mock.sentinel.result) as regrid:
            result = regridder(src)

        self.assertEqual(regrid.call_count, 1)
        _, args, kwargs = regrid.mock_calls[0]

        self.assertEqual(args[0], src)
        self.assertEqual(self.extract_grid(args[1]),
                         self.extract_grid(target_grid))
        self.assertEqual(kwargs, {'mdtol': mdtol})
        self.assertIs(result, mock.sentinel.result)

    def test_default(self):
        self.check_mdtol()

    def test_specified_mdtol(self):
        self.check_mdtol(0.5)

    def test_invalid_high_mdtol(self):
        src, target = self.grids()
        msg = 'mdtol must be in range 0 - 1'
        with self.assertRaisesRegexp(ValueError, msg):
            AreaWeightedRegridder(src, target, mdtol=1.2)

    def test_invalid_low_mdtol(self):
        src, target = self.grids()
        msg = 'mdtol must be in range 0 - 1'
        with self.assertRaisesRegexp(ValueError, msg):
            AreaWeightedRegridder(src, target, mdtol=-0.2)

    def test_mismatched_src_coord_systems(self):
        src = Cube(np.zeros((3, 4)))
        cs = GeogCS(6543210)
        lat = DimCoord(np.arange(3), 'latitude', coord_system=cs)
        lon = DimCoord(np.arange(4), 'longitude')
        src.add_dim_coord(lat, 0)
        src.add_dim_coord(lon, 1)
        target = mock.Mock()
        with self.assertRaises(ValueError):
            AreaWeightedRegridder(src, target)


if __name__ == '__main__':
    tests.main()
