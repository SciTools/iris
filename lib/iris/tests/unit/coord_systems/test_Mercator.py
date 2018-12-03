# (C) British Crown Copyright 2018, Met Office
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
"""Unit tests for the :class:`iris.coord_systems.Mercator` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import cartopy.crs as ccrs
from iris.coord_systems import GeogCS, Mercator


class Test_Mercator__basics(tests.IrisTest):
    def setUp(self):
        self.tm = Mercator(longitude_of_projection_origin=90.0,
                           ellipsoid=GeogCS(6377563.396, 6356256.909))

    def test_construction(self):
        self.assertXMLElement(self.tm, ("coord_systems", "Mercator.xml"))

    def test_repr(self):
        expected = ("Mercator(longitude_of_projection_origin=90.0, "
                    "ellipsoid=GeogCS(semi_major_axis=6377563.396, "
                    "semi_minor_axis=6356256.909), "
                    "standard_parallel=0.0)")
        self.assertEqual(expected, repr(self.tm))


class Test_Mercator__as_cartopy_crs(tests.IrisTest):
    def test_simple(self):
        # Check that a projection set up with all the defaults is correctly
        # converted to a cartopy CRS.
        merc_cs = Mercator()
        res = merc_cs.as_cartopy_crs()
        # expected = ccrs.Mercator(globe=ccrs.Globe())
        expected = ccrs.Mercator(globe=ccrs.Globe(), latitude_true_scale=0.0)
        self.assertEqual(res, expected)

    def test_extra_kwargs(self):
        # Check that a projection with non-default values is correctly
        # converted to a cartopy CRS.
        longitude_of_projection_origin = 90.0
        true_scale_lat = 14.0
        ellipsoid = GeogCS(semi_major_axis=6377563.396,
                           semi_minor_axis=6356256.909)

        merc_cs = Mercator(
            longitude_of_projection_origin,
            ellipsoid=ellipsoid,
            standard_parallel=true_scale_lat)

        expected = ccrs.Mercator(
            central_longitude=longitude_of_projection_origin,
            globe=ccrs.Globe(semimajor_axis=6377563.396,
                             semiminor_axis=6356256.909, ellipse=None),
            latitude_true_scale=true_scale_lat)

        res = merc_cs.as_cartopy_crs()
        self.assertEqual(res, expected)


class Test_as_cartopy_projection(tests.IrisTest):
    def test_simple(self):
        # Check that a projection set up with all the defaults is correctly
        # converted to a cartopy projection.
        merc_cs = Mercator()
        res = merc_cs.as_cartopy_projection()
        expected = ccrs.Mercator(globe=ccrs.Globe(), latitude_true_scale=0.0)
        self.assertEqual(res, expected)

    def test_extra_kwargs(self):
        longitude_of_projection_origin = 90.0
        true_scale_lat = 14.0
        ellipsoid = GeogCS(semi_major_axis=6377563.396,
                           semi_minor_axis=6356256.909)

        merc_cs = Mercator(
            longitude_of_projection_origin,
            ellipsoid=ellipsoid,
            standard_parallel=true_scale_lat)

        expected = ccrs.Mercator(
            central_longitude=longitude_of_projection_origin,
            globe=ccrs.Globe(semimajor_axis=6377563.396,
                             semiminor_axis=6356256.909, ellipse=None),
            latitude_true_scale=true_scale_lat)

        res = merc_cs.as_cartopy_projection()
        self.assertEqual(res, expected)


if __name__ == "__main__":
    tests.main()
