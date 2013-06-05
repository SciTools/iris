# (C) British Crown Copyright 2010 - 2013, Met Office
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


from __future__ import division

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import logging

import iris.coords
import iris.unit

from iris.coord_systems import *


logger = logging.getLogger('tests')


def osgb():
    return TransverseMercator(latitude_of_projection_origin=49, longitude_of_central_meridian=-2,
                              false_easting=-400, false_northing=100,
                              scale_factor_at_central_meridian=0.9996012717,
                              ellipsoid=GeogCS(6377563.396, 6356256.909))



class TestCoordSystemSame(tests.IrisTest):

    def setUp(self):
        self.cs1 = iris.coord_systems.GeogCS(6371229)
        self.cs2 = iris.coord_systems.GeogCS(6371229)
        self.cs3 = iris.coord_systems.RotatedGeogCS(30, 30, ellipsoid=GeogCS(6371229))

    def test_simple(self):
        a = self.cs1
        b = self.cs2
        self.assertEquals(a, b)

    def test_different_class(self):
        a = self.cs1
        b = self.cs3
        self.assertNotEquals(a, b)

    def test_different_public_attributes(self):
        a = self.cs1
        b = self.cs2
        a.foo = 'a'

        # check that that attribute was added (just in case)
        self.assertEqual(a.foo, 'a')

        # a and b should not be the same
        self.assertNotEquals(a, b)

        # a and b should be the same
        b.foo = 'a'
        self.assertEquals(a, b)

        b.foo = 'b'
        # a and b should not be the same
        self.assertNotEquals(a, b)


class Test_CoordSystem_xml_element(tests.IrisTest):
    def test_rotated(self):
        cs = RotatedGeogCS(30, 40, ellipsoid=GeogCS(6371229))
        self.assertXMLElement(cs, ("coord_systems", "CoordSystem_xml_element.xml"))


class Test_GeogCS_construction(tests.IrisTest):
    # Test Ellipsoid constructor
    # Don't care about testing the units, it has no logic specific to this class.

    def test_sphere_param(self):
        cs = GeogCS(6543210)
        self.assertXMLElement(cs, ("coord_systems", "GeogCS_init_sphere.xml"))

    def test_no_major(self):
        cs = GeogCS(semi_minor_axis=6500000, inverse_flattening=151.42814163388104)
        self.assertXMLElement(cs, ("coord_systems", "GeogCS_init_no_major.xml"))

    def test_no_minor(self):
        cs = GeogCS(semi_major_axis=6543210, inverse_flattening=151.42814163388104)
        self.assertXMLElement(cs, ("coord_systems", "GeogCS_init_no_minor.xml"))

    def test_no_invf(self):
        cs = GeogCS(semi_major_axis=6543210, semi_minor_axis=6500000)
        self.assertXMLElement(cs, ("coord_systems", "GeogCS_init_no_invf.xml"))

    def test_invalid_ellipsoid_params(self):
        # no params
        with self.assertRaises(ValueError):
            GeogCS()

        # over specified
        with self.assertRaises(ValueError):
            GeogCS(6543210, 6500000, 151.42814163388104)

        # under specified
        with self.assertRaises(ValueError):
            GeogCS(None, 6500000, None)
        with self.assertRaises(ValueError):
            GeogCS(None, None, 151.42814163388104)


class Test_GeogCS_repr(tests.IrisTest):
    def test_repr(self):
        cs = GeogCS(6543210, 6500000)
        expected = "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0)"
        self.assertEqual(expected, repr(cs))

class Test_GeogCS_str(tests.IrisTest):
    def test_str(self):
        cs = GeogCS(6543210, 6500000)
        expected = "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0)"
        self.assertEqual(expected, str(cs))


class Test_RotatedGeogCS_construction(tests.IrisTest):
    def test_init(self):
        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50, ellipsoid=GeogCS(6371229))
        self.assertXMLElement(rcs, ("coord_systems", "RotatedGeogCS_init.xml"))

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        self.assertXMLElement(rcs, ("coord_systems", "RotatedGeogCS_init_a.xml"))

        rcs = RotatedGeogCS(30, 40)
        self.assertXMLElement(rcs, ("coord_systems", "RotatedGeogCS_init_b.xml"))


class Test_RotatedGeogCS_repr(tests.IrisTest):
    def test_repr(self):
        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50, ellipsoid=GeogCS(6371229))
        expected = "RotatedGeogCS(30.0, 40.0, "\
                    "north_pole_grid_longitude=50.0, ellipsoid=GeogCS(6371229.0))"
        self.assertEqual(expected, repr(rcs))

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        expected = "RotatedGeogCS(30.0, 40.0, north_pole_grid_longitude=50.0)"
        self.assertEqual(expected, repr(rcs))

        rcs = RotatedGeogCS(30, 40)
        expected = "RotatedGeogCS(30.0, 40.0)"
        self.assertEqual(expected, repr(rcs))


class Test_RotatedGeogCS_str(tests.IrisTest):
    def test_str(self):
        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50, ellipsoid=GeogCS(6371229))
        expected = "RotatedGeogCS(30.0, 40.0, "\
                    "north_pole_grid_longitude=50.0, ellipsoid=GeogCS(6371229.0))"
        self.assertEqual(expected, str(rcs))

        rcs = RotatedGeogCS(30, 40, north_pole_grid_longitude=50)
        expected = "RotatedGeogCS(30.0, 40.0, north_pole_grid_longitude=50.0)"
        self.assertEqual(expected, str(rcs))

        rcs = RotatedGeogCS(30, 40)
        expected = "RotatedGeogCS(30.0, 40.0)"
        self.assertEqual(expected, str(rcs))


class Test_TransverseMercator_construction(tests.IrisTest):
    def test_osgb(self):
        tm = osgb()
        self.assertXMLElement(tm, ("coord_systems", "TransverseMercator_osgb.xml"))


class Test_TransverseMercator_repr(tests.IrisTest):
    def test_osgb(self):
        tm = osgb()
        expected = "TransverseMercator(latitude_of_projection_origin=49.0, longitude_of_central_meridian=-2.0, "\
                    "false_easting=-400.0, false_northing=100.0, scale_factor_at_central_meridian=0.9996012717, "\
                    "ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909))"
        self.assertEqual(expected, repr(tm))


if __name__ == "__main__":
    tests.main()
