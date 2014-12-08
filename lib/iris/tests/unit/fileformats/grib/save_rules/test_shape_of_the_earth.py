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
"""
Unit tests for
:meth:`iris.fileformats.grib._save_rules.shape_of_the_earth`.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coord_systems import GeogCS, TransverseMercator, OSGB
from iris.exceptions import TranslationError
from iris.tests.unit.fileformats.grib.save_rules import GdtTestMixin

from iris.fileformats.grib._save_rules import shape_of_the_earth


class Test(tests.IrisTest, GdtTestMixin):
    def setUp(self):
        GdtTestMixin.setUp(self)

    def _spherical_earth_test_common(self, radius):
        self._check_key('scaleFactorOfRadiusOfSphericalEarth', 0)
        self._check_key('scaledValueOfRadiusOfSphericalEarth', radius)

    def _oblate_spheroid_earth_test_common(self, semi_major_axis,
                                           semi_minor_axis):
        self._check_key('scaleFactorOfEarthMajorAxis', 0)
        self._check_key('scaledValueOfEarthMajorAxis', semi_major_axis)
        self._check_key('scaleFactorOfEarthMinorAxis', 0)
        self._check_key('scaledValueOfEarthMinorAxis', semi_minor_axis)

    def test_radius_of_earth_6367470(self):
        # Test setting shapeOfTheEarth = 0
        radius = 6367470
        cs = GeogCS(semi_major_axis=radius)
        test_cube = self._make_test_cube(cs=cs)
        shape_of_the_earth(test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 0)
        self._spherical_earth_test_common(radius)

    def test_radius_of_earth_6371229(self):
        # Test setting shapeOfTheEarth = 6
        radius = 6371229
        cs = GeogCS(semi_major_axis=radius)
        test_cube = self._make_test_cube(cs=cs)
        shape_of_the_earth(test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 6)
        self._spherical_earth_test_common(radius)

    def test_spherical_earth(self):
        # Test setting shapeOfTheEarth = 1
        radius = 1.23
        cs = GeogCS(semi_major_axis=radius)
        test_cube = self._make_test_cube(cs=cs)
        shape_of_the_earth(test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 1)
        self._spherical_earth_test_common(radius)

    def test_oblate_spheroid_earth(self):
        # Test setting shapeOfTheEarth = 7
        semi_major_axis = 1.456
        semi_minor_axis = 1.123
        cs = GeogCS(semi_major_axis=semi_major_axis,
                    semi_minor_axis=semi_minor_axis)
        test_cube = self._make_test_cube(cs=cs)
        shape_of_the_earth(test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 7)
        self._oblate_spheroid_earth_test_common(semi_major_axis,
                                                semi_minor_axis)

    def test_OSGB(self):
        # Test setting shapeOfTheEarth = 9
        cs = OSGB()
        # The following are fixed for the OSGB coord system.
        semi_major_axis, semi_minor_axis = (6377563.396, 6356256.909)
        test_cube = self._make_test_cube(cs=cs)
        shape_of_the_earth(test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 9)
        self._oblate_spheroid_earth_test_common(semi_major_axis,
                                                semi_minor_axis)

    def test_TransverseMercator_spherical(self):
        # Test setting shapeOfTheEarth = 1 with a non-GeogCS coord system.
        cs = TransverseMercator(49, -2, 400000, -100000, 0.9996012717,
                                ellipsoid=GeogCS(6377563.396))
        radius = 6377563.396
        test_cube = self._make_test_cube(cs=cs)
        shape_of_the_earth(test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 1)
        self._spherical_earth_test_common(radius)


if __name__ == "__main__":
    tests.main()
