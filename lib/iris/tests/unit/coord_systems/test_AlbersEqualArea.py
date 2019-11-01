# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.coord_systems.AlbersEqualArea` class.

"""

from __future__ import absolute_import, division, print_function
from six.moves import filter, input, map, range, zip  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import cartopy.crs as ccrs
from iris.coord_systems import GeogCS, AlbersEqualArea


class Test_as_cartopy_crs(tests.IrisTest):
    def setUp(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_central_meridian = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.standard_parallels = (-18.0, -36.0)
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.aea_cs = AlbersEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_central_meridian,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            ellipsoid=self.ellipsoid,
        )

    def test_crs_creation(self):
        res = self.aea_cs.as_cartopy_crs()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.AlbersEqualArea(
            self.longitude_of_central_meridian,
            self.latitude_of_projection_origin,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            globe=globe,
        )
        self.assertEqual(res, expected)


class Test_as_cartopy_projection(tests.IrisTest):
    def setUp(self):
        self.latitude_of_projection_origin = 0.0
        self.longitude_of_central_meridian = 0.0
        self.semi_major_axis = 6377563.396
        self.semi_minor_axis = 6356256.909
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.standard_parallels = (-18.0, -36.0)
        self.ellipsoid = GeogCS(self.semi_major_axis, self.semi_minor_axis)
        self.aea_cs = AlbersEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_central_meridian,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            ellipsoid=self.ellipsoid,
        )

    def test_projection_creation(self):
        res = self.aea_cs.as_cartopy_projection()
        globe = ccrs.Globe(
            semimajor_axis=self.semi_major_axis,
            semiminor_axis=self.semi_minor_axis,
            ellipse=None,
        )
        expected = ccrs.AlbersEqualArea(
            self.latitude_of_projection_origin,
            self.longitude_of_central_meridian,
            self.false_easting,
            self.false_northing,
            self.standard_parallels,
            globe=globe,
        )
        self.assertEqual(res, expected)


if __name__ == "__main__":
    tests.main()
