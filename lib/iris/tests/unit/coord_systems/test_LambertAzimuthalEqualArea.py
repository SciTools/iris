# (C) British Crown Copyright 2015 - 2016, Met Office
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
Unit tests for the :class:`iris.coord_systems.LambertAzimuthalEqualArea` class.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import cartopy.crs as ccrs
from iris.coord_systems import GeogCS, LambertAzimuthalEqualArea


class Test_as_cartopy_crs(tests.IrisTest):
    def setUp(self):
        self.latitude_of_projection_origin = 90.0
        self.longitude_of_projection_origin = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.laea_cs = LambertAzimuthalEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            ellipsoid=self.ellipsoid)

    def test_crs_creation(self):
        res = self.laea_cs.as_cartopy_crs()
        globe = ccrs.Globe(semimajor_axis=self.semi_major_axis,
                           semiminor_axis=self.semi_minor_axis,
                           ellipse=None)
        expected = ccrs.LambertAzimuthalEqualArea(
            self.longitude_of_projection_origin,
            self.latitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            globe=globe)
        self.assertEqual(res, expected)


class Test_as_cartopy_projection(tests.IrisTest):
    def setUp(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_projection_origin = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.laea_cs = LambertAzimuthalEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            ellipsoid=self.ellipsoid)

    def test_projection_creation(self):
        res = self.laea_cs.as_cartopy_projection()
        globe = ccrs.Globe(semimajor_axis=self.semi_major_axis,
                           semiminor_axis=self.semi_minor_axis,
                           ellipse=None)
        expected = ccrs.LambertAzimuthalEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            globe=globe)
        self.assertEqual(res, expected)


if __name__ == '__main__':
    tests.main()
