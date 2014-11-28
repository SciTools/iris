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
:meth:`iris.fileformats.grib._save_rules.grid_definition_template_1`.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.coord_systems import GeogCS, RotatedGeogCS
from iris.exceptions import TranslationError
from iris.tests.unit.fileformats.grib.save_rules import GdtTestMixin

from iris.fileformats.grib._save_rules import grid_definition_template_1

TARGET_MODULE = 'iris.fileformats.grib._save_rules'


class Test(tests.IrisTest, GdtTestMixin):
    def setUp(self):
        GdtTestMixin.setUp(self)

    def _default_coord_system(self):
        self.mock_ellipsoid = mock.Mock(spec=GeogCS,
                                        semi_major_axis=1.0,
                                        semi_minor_axis=1.0,
                                        inverse_flattening=None,
                                        longitude_of_prime_meridian=0)
        self.mock_cs = mock.Mock(spec=RotatedGeogCS,
                                 grid_north_pole_latitude=90.0,
                                 grid_north_pole_longitude=0.0,
                                 north_pole_grid_longitude=0.0,
                                 ellipsoid=self.mock_ellipsoid)
        return self.mock_cs

    def test__template_number(self):
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key('gridDefinitionTemplateNumber', 1)

    def test__shape_of_earth_spherical(self):
        cs = self.mock_x_coord.coord_system.ellipsoid
        cs.inverse_flattening = 0
        cs.semi_major_axis = 1.23
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 1)
        self._check_key('scaleFactorOfRadiusOfSphericalEarth', 0)
        self._check_key('scaledValueOfRadiusOfSphericalEarth', 1.23)

    def test__shape_of_earth_flattened(self):
        cs = self.mock_x_coord.coord_system.ellipsoid
        cs.inverse_flattening = 7777  # This is used just as a flag.
        cs.semi_major_axis = 1.456
        cs.semi_minor_axis = 1.123
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 7)
        self._check_key('scaleFactorOfEarthMajorAxis', 0)
        self._check_key('scaledValueOfEarthMajorAxis', 1.456)
        self._check_key('scaleFactorOfEarthMinorAxis', 0)
        self._check_key('scaledValueOfEarthMinorAxis', 1.123)

    def test__grid_shape(self):
        self._set_coords(x_points=np.arange(13), y_points=np.arange(6))
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key('Ni', 13)
        self._check_key('Nj', 6)

    def test__grid_points(self):
        self._set_coords(x_points=[1, 3, 5, 7], y_points=[4, 9])
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key("longitudeOfFirstGridPoint", 1000000)
        self._check_key("longitudeOfLastGridPoint", 7000000)
        self._check_key("latitudeOfFirstGridPoint", 4000000)
        self._check_key("latitudeOfLastGridPoint", 9000000)
        self._check_key("DxInDegrees", 2.0)
        self._check_key("DyInDegrees", 5.0)

    def test__scanmode(self):
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key('iScansPositively', 1)
        self._check_key('jScansPositively', 1)

    def test__scanmode_reverse(self):
        self._set_coords(x_points=np.arange(7, 0, -1))
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key('iScansPositively', 0)
        self._check_key('jScansPositively', 1)

    def test__rotated_pole(self):
        cs = self.mock_cs
        cs.grid_north_pole_latitude = 75.3
        cs.grid_north_pole_longitude = 54.321
        grid_definition_template_1(self.mock_cube, self.mock_grib)
        self._check_key("latitudeOfSouthernPole", -75300000)
        self._check_key("longitudeOfSouthernPole", 234321000)
        self._check_key("angleOfRotation", 0)


if __name__ == "__main__":
    tests.main()
