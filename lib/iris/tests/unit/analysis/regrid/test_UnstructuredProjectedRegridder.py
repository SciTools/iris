# (C) British Crown Copyright 2016, Met Office
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
"""Unit tests for :class:`iris.analysis._regrid.ProjectedUnstructuredRegridder`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import cartopy.crs as ccrs

from iris.analysis._regrid import ProjectedUnstructuredRegridder as Regridder
from iris.aux_factory import HybridHeightFactory
from iris.coord_systems import GeogCS, OSGB
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import mock
from iris.tests.stock import global_pp, lat_lon_cube, realistic_4d


RESULT_DIR = ('analysis', 'regrid')

# Convenience to access Regridder static method.
regrid = Regridder._regrid


class Test__regrid_xy_dim_position(tests.IrisTest):
    def setUp(self):
        self.shape = (3, 2, 4)
        self.src_data = np.zeros(self.shape)
        self.crs = GeogCS(6371229)
        self.tgt_x_coord = DimCoord(np.arange(5), standard_name='longitude',
                                    units='degrees', coord_system=self.crs)
        self.tgt_y_coord = DimCoord(np.arange(6), standard_name='latitude',
                                    units='degrees', coord_system=self.crs)

    def check_call(self, xy_dim):
        src_x_coord = AuxCoord(np.arange(self.shape[xy_dim]),
                               standard_name='longitude',
                               units='degrees', coord_system=self.crs)
        src_y_coord = AuxCoord(np.arange(self.shape[xy_dim])*2,
                               standard_name='longitude',
                               units='degrees', coord_system=self.crs)

        result = regrid(self.src_data, xy_dim, src_x_coord, src_y_coord,
                        self.tgt_x_coord, self.tgt_y_coord,
                        self.crs.as_cartopy_projection())

    def test_xy_dim_first_position(self):
        xy_dim = 0
        expected_shape = (6, 5, 2, 4)
        result = self.check_call(xy_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_xy_dim_middle_position(self):
        xy_dim = 1
        expected_shape = (3, 6, 5, 4)
        result = self.check_call(xy_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_xy_dim_last_position(self):
        xy_dim = 2
        expected_shape = (3, 2, 6, 5)
        result = self.check_call(xy_dim)
        self.assertEqual(result.shape, expected_shape)


class Test__regrid_projection(tests.IrisTest):
    def setUp(self):
        self.src_data = np.arange(7)
        self.crs = GeogCS(6371229)
        self.src_x_coord = AuxCoord([0.4, 0.8, 1.5, 2.7, 3., 3.6, 4],
                                    standard_name='longitude',
                                    units='degrees', coord_system=self.crs)
        self.src_y_coord = AuxCoord([0.3, 3., 1.2, 0, 2, 2.4, 2.8],
                                    standard_name='latitude',
                                    units='degrees', coord_system=self.crs)

        self.tgt_x_coord = DimCoord(np.arange(5), standard_name='longitude',
                                    units='degrees', coord_system=self.crs)
        self.tgt_y_coord = DimCoord(np.arange(4), standard_name='latitude',
                                    units='degrees', coord_system=self.crs)

    def check_call(self, projection):
        xy_dim = 0
        result = regrid(self.src_data, xy_dim, self.src_x_coord, self.src_y_coord,
                        self.tgt_x_coord, self.tgt_y_coord,
                        projection)

    def test_sinusoidal(self):
        globe = self.crs.as_cartopy_globe()
        projection = ccrs.Sinusoidal(globe=globe)
        expected = np.array([[0, 0, 2, 3, 6],
                             [0, 2, 2, 4, 6],
                             [1, 2, 2, 4, 6],
                             [1, 1, 2, 4, 5]])
        result = self.check_call(projection)
        self.assertArrayEqual(result, expected)


if __name__ == '__main__':
    tests.main()
