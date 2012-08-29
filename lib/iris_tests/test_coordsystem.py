# (C) British Crown Copyright 2010 - 2012, Met Office
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
    return TransverseMercator(GeogCS(6377563.396, 6356256.909, 299.3249646, "m"), 
                              origin=(49,-2), false_origin=(-400,100),
                              scale_factor=0.9996012717)



class TestCoordSystemSame(tests.IrisTest):

    def setUp(self):
        self.cs1 = iris.coord_systems.GeogCS()
        self.cs2 = iris.coord_systems.GeogCS()
        self.cs3 = iris.coord_systems.RotatedGeogCS(grid_north_pole=(30,30))

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
        cs = RotatedGeogCS(grid_north_pole=(30,40))
        self.assertXMLElement(cs, tests.get_result_path(("coord_systems", "CoordSystem_xml_element.xml")))


class Test_GeogCS_init(tests.IrisTest):
    # Test Ellipsoid constructor
    # Don't care about testing the units, it has no logic specific to this class.
    
    def test_no_params(self):
        cs = GeogCS()
        self.assertXMLElement(cs, tests.get_result_path(("coord_systems", "GeogCS_init_no_param.xml")))

    def test_sphere_param(self):
        cs = GeogCS(6543210, units="m")
        self.assertXMLElement(cs, tests.get_result_path(("coord_systems", "GeogCS_init_sphere.xml")))

    def test_all_ellipsoid_params(self):
        cs = GeogCS(6543210, 6500000, 151.42814163388104, "m")
        self.assertXMLElement(cs, tests.get_result_path(("coord_systems", "GeogCS_init_all_ellipsoid.xml")))

    def test_no_major(self):
        cs = GeogCS(semi_minor_axis=6500000, inverse_flattening=151.42814163388104, units="m")
        self.assertXMLElement(cs, tests.get_result_path(("coord_systems", "GeogCS_init_no_major.xml")))
    
    def test_no_minor(self):
        cs = GeogCS(semi_major_axis=6543210, inverse_flattening=151.42814163388104, units="m")
        self.assertXMLElement(cs, tests.get_result_path(("coord_systems", "GeogCS_init_no_minor.xml")))
    
    def test_no_invf(self):
        cs = GeogCS(semi_major_axis=6543210, semi_minor_axis=6500000, units="m")
        self.assertXMLElement(cs, tests.get_result_path(("coord_systems", "GeogCS_init_no_invf.xml")))

    def test_units(self):
        # Just make sure they get conveted to a unit, not overly concerned about testing this param.
        cs = GeogCS(6543210, units="m")
        self.assertEqual(cs.units, iris.unit.Unit("m"))


class Test_GeogCS_repr(tests.IrisTest):
    def test_repr(self): 
        cs = GeogCS(6543210, 6500000, 151.42814163388104, "m")
        expected = "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0, inverse_flattening=151.42814163388104, units='Unit('m')', longitude_of_prime_meridian=0.0)"
        self.assertEqual(expected, repr(cs))

class Test_GeogCS_str(tests.IrisTest):
    def test_str(self): 
        cs = GeogCS(6543210, 6500000, 151.42814163388104, "m")
        expected = "GeogCS(semi_major_axis=6543210.0, semi_minor_axis=6500000.0, inverse_flattening=151.428141634, units=m, longitude_of_prime_meridian=0.0)"
        self.assertEqual(expected, str(cs))


class Test_GeogCS_to_cartopy(tests.IrisTest):
    def test_to_cartopy(self):
        cs = GeogCS()
        self.assertRaises(NotImplementedError, cs._to_cartopy)


class Test_RotatedGeogCS_init(tests.IrisTest):
    def test_init(self):
        rcs = RotatedGeogCS(grid_north_pole=(30,40), north_pole_lon=50)
        self.assertXMLElement(rcs, tests.get_result_path(("coord_systems", "RotatedGeogCS_init.xml")))


class Test_RotatedGeogCS_repr(tests.IrisTest):
    def test_repr(self): 
        rcs = RotatedGeogCS(grid_north_pole=(30,40), north_pole_lon=50)
        expected = "RotatedGeogCS(semi_major_axis=6371229.0, semi_minor_axis=6371229.0, inverse_flattening=0.0, units='Unit('m')', longitude_of_prime_meridian=0.0, grid_north_pole=GeoPos(30.0, 40.0), north_pole_lon=50.0)"
        self.assertEqual(expected, repr(rcs))

class Test_RotatedGeogCS_str(tests.IrisTest):
    def test_str(self): 
        rcs = RotatedGeogCS(grid_north_pole=(30,40), north_pole_lon=50)
        expected = "RotatedGeogCS(semi_major_axis=6371229.0, semi_minor_axis=6371229.0, inverse_flattening=0.0, units=m, longitude_of_prime_meridian=0.0, grid_north_pole=GeoPos(30.0, 40.0), north_pole_lon=50.0)"
        self.assertEqual(expected, str(rcs))


class Test_RotatedGeogCS_from_geocs(tests.IrisTest):
    def test_sphere(self):
        cs = GeogCS()
        rcs = RotatedGeogCS.from_geocs(cs, grid_north_pole=(30,40))
        self.assertXMLElement(rcs, tests.get_result_path(("coord_systems", "RotatedGeogCS_from_geocs.xml")))

class Test_RotatedGeogCS_tocartopy(tests.IrisTest):
    def test_to_cartopy(self):
        rcs = GeogCS()
        self.assertRaises(NotImplementedError, rcs._to_cartopy)


class Test_TransverseMercator_init(tests.IrisTest):
    def test_osgb(self):
        tm = osgb()
        self.assertXMLElement(tm, tests.get_result_path(("coord_systems", "TransverseMercator_osgb.xml")))


class Test_TransverseMercator_repr(tests.IrisTest):
    def test_osgb(self):
        tm = osgb()
        expected = "TransverseMercator(origin=GeoPos(49.0, -2.0), false_origin=(-400.0, 100.0), scale_factor=0.9996012717, geos=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909, inverse_flattening=299.3249646, units='Unit('m')', longitude_of_prime_meridian=0.0))"
        self.assertEqual(expected, repr(tm))


class Test_TransverseMercator_tocartopy(tests.IrisTest):
    def test_to_cartopy(self):
        tm = osgb()
        self.assertRaises(NotImplementedError, tm._to_cartopy)
    

if __name__ == "__main__":
    tests.main()
