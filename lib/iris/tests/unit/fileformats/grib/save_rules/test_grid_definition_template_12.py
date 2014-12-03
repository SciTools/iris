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
:meth:`iris.fileformats.grib._save_rules.grid_definition_template_12`.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coord_systems import GeogCS, TransverseMercator
from iris.exceptions import TranslationError
from iris.tests.unit.fileformats.grib.save_rules import GdtTestMixin

from iris.fileformats.grib._save_rules import grid_definition_template_12


class Test(tests.IrisTest, GdtTestMixin):
    def setUp(self):
        self.default_ellipsoid = GeogCS(semi_major_axis=6377563.396,
                                        semi_minor_axis=6356256.909)
        self.default_cs = self._default_coord_system()
        self.test_cube = self._make_test_cube(cs=self.default_cs)
        GdtTestMixin.setUp(self)

    def _default_coord_system(self):
        # This defines an OSGB coord system.
        cs = TransverseMercator(latitude_of_projection_origin=49.0,
                                longitude_of_central_meridian=-2.0,
                                false_easting=400000.0,
                                false_northing=-100000.0,
                                scale_factor_at_central_meridian=0.9996012717,
                                ellipsoid=self.default_ellipsoid)
        return cs

    def test__template_number(self):
        grid_definition_template_12(self.test_cube, self.mock_grib)
        self._check_key('gridDefinitionTemplateNumber', 12)

    def test__shape_of_earth(self):
        grid_definition_template_12(self.test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 7)
        self._check_key('scaleFactorOfEarthMajorAxis', 0)
        self._check_key('scaledValueOfEarthMajorAxis', 6377563.396)
        self._check_key('scaleFactorOfEarthMinorAxis', 0)
        self._check_key('scaledValueOfEarthMinorAxis', 6356256.909)

    def test__grid_shape(self):
        test_cube = self._make_test_cube(x_points=np.arange(13),
                                         y_points=np.arange(6),
                                         cs=self.default_cs)
        grid_definition_template_12(test_cube, self.mock_grib)
        self._check_key('Ni', 13)
        self._check_key('Nj', 6)

    def test__grid_points(self):
        test_cube = self._make_test_cube(x_points=[1, 3, 5, 7],
                                         y_points=[4, 9],
                                         cs=self.default_cs)
        grid_definition_template_12(test_cube, self.mock_grib)
        self._check_key("X1", 100)
        self._check_key("X2", 700)
        self._check_key("Y1", 400)
        self._check_key("Y2", 900)
        self._check_key("Di", 200.0)
        self._check_key("Dj", 500.0)

    def test__template_specifics(self):
        grid_definition_template_12(self.test_cube, self.mock_grib)
        self._check_key("latitudeOfReferencePoint", 49000000.0)
        self._check_key("longitudeOfReferencePoint", -2000000.0)
        self._check_key("XR", 40000000.0)
        self._check_key("YR", -10000000.0)
        self._check_key("scaleFactorAtReferencePoint", 0.9996012717)

    def test__scanmode(self):
        grid_definition_template_12(self.test_cube, self.mock_grib)
        self._check_key('iScansPositively', 1)
        self._check_key('jScansPositively', 1)

    def test__scanmode_reverse(self):
        test_cube = self._make_test_cube(x_points=np.arange(7, 0, -1))
        grid_definition_template_12(test_cube, self.mock_grib)
        self._check_key('iScansPositively', 0)
        self._check_key('jScansPositively', 1)


if __name__ == "__main__":
    tests.main()
