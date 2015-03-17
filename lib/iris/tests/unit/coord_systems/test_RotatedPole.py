# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for the :class:`iris.coord_systems.RotatedPole` class."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import cartopy.crs as ccrs
from iris.coord_systems import GeogCS, RotatedGeogCS


class Test_init(tests.IrisTest):
    def setUp(self):
        self.pole_lon = 171.77
        self.pole_lat = 49.55
        self.rotation_about_new_pole = 180.0
        self.rp_crs = RotatedGeogCS(self.pole_lat, self.pole_lon,
                                    self.rotation_about_new_pole)

    def test_crs_creation(self):
        self.assertEqual(self.pole_lon, self.rp_crs.grid_north_pole_longitude)
        self.assertEqual(self.pole_lat, self.rp_crs.grid_north_pole_latitude)
        self.assertEqual(self.rotation_about_new_pole,
                         self.rp_crs.north_pole_grid_longitude)

    def test_as_cartopy_crs(self):
        accrs = self.rp_crs.as_cartopy_crs()
        expected = ccrs.RotatedGeodetic(self.pole_lon, self.pole_lat,
                                        self.rotation_about_new_pole)
        self.assertEqual(sorted(accrs.proj4_init.split(' +')),
                         sorted(expected.proj4_init.split(' +')))

    def test_as_cartopy_projection(self):
        accrsp = self.rp_crs.as_cartopy_projection()
        expected = ccrs.RotatedPole(self.pole_lon, self.pole_lat,
                                    self.rotation_about_new_pole)
        self.assertEqual(sorted(accrsp.proj4_init.split(' +')),
                         sorted(expected.proj4_init.split(' +')))

if __name__ == '__main__':
    tests.main()
