# (C) British Crown Copyright 2017, Met Office
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
:meth:`iris.fileformats.grib._save_rules.grid_definition_template_30`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris.coords
from iris.coord_systems import GeogCS, LambertConformal
from iris.exceptions import TranslationError
from iris.fileformats.grib._save_rules import grid_definition_template_30
from iris.tests.unit.fileformats.grib.save_rules import GdtTestMixin


class FakeGribError(Exception):
    pass


class Test(tests.IrisTest, GdtTestMixin):
    def setUp(self):
        self.default_ellipsoid = GeogCS(semi_major_axis=6377563.396,
                                        semi_minor_axis=6356256.909)
        self.test_cube = self._make_test_cube()

        GdtTestMixin.setUp(self)

    def _make_test_cube(self, cs=None, x_points=None, y_points=None):
        # Create a cube with given properties, or minimal defaults.
        if cs is None:
            cs = self._default_coord_system()
        if x_points is None:
            x_points = self._default_x_points()
        if y_points is None:
            y_points = self._default_y_points()

        x_coord = iris.coords.DimCoord(x_points, 'projection_x_coordinate',
                                       units='m', coord_system=cs)
        y_coord = iris.coords.DimCoord(y_points, 'projection_y_coordinate',
                                       units='m', coord_system=cs)
        test_cube = iris.cube.Cube(np.zeros((len(y_points), len(x_points))))
        test_cube.add_dim_coord(y_coord, 0)
        test_cube.add_dim_coord(x_coord, 1)
        return test_cube

    def _default_coord_system(self):
        return LambertConformal(central_lat=39.0, central_lon=-96.0,
                                false_easting=0.0, false_northing=0.0,
                                secant_latitudes=(33, 45),
                                ellipsoid=self.default_ellipsoid)

    def test__template_number(self):
        grid_definition_template_30(self.test_cube, self.mock_grib)
        self._check_key('gridDefinitionTemplateNumber', 30)

    def test__shape_of_earth(self):
        grid_definition_template_30(self.test_cube, self.mock_grib)
        self._check_key('shapeOfTheEarth', 7)
        self._check_key('scaleFactorOfEarthMajorAxis', 0)
        self._check_key('scaledValueOfEarthMajorAxis', 6377563.396)
        self._check_key('scaleFactorOfEarthMinorAxis', 0)
        self._check_key('scaledValueOfEarthMinorAxis', 6356256.909)

    def test__grid_shape(self):
        test_cube = self._make_test_cube(x_points=np.arange(13),
                                         y_points=np.arange(6))
        grid_definition_template_30(test_cube, self.mock_grib)
        self._check_key('Nx', 13)
        self._check_key('Ny', 6)

    def test__grid_points(self):
        test_cube = self._make_test_cube(x_points=[1e6, 3e6, 5e6, 7e6],
                                         y_points=[4e6, 9e6])
        grid_definition_template_30(test_cube, self.mock_grib)
        self._check_key("latitudeOfFirstGridPoint", 71676530)
        self._check_key("longitudeOfFirstGridPoint", 287218188)
        self._check_key("Dx", 2e9)
        self._check_key("Dy", 5e9)

    def test__template_specifics(self):
        grid_definition_template_30(self.test_cube, self.mock_grib)
        self._check_key("LaD", 39e6)
        self._check_key("LoV", 264e6)
        self._check_key("Latin1", 33e6)
        self._check_key("Latin2", 45e6)
        self._check_key("latitudeOfSouthernPole", 0)
        self._check_key("longitudeOfSouthernPole", 0)

    def test__scanmode(self):
        grid_definition_template_30(self.test_cube, self.mock_grib)
        self._check_key('iScansPositively', 1)
        self._check_key('jScansPositively', 1)

    def test__scanmode_reverse(self):
        test_cube = self._make_test_cube(x_points=np.arange(7e6, 0, -1e6))
        grid_definition_template_30(test_cube, self.mock_grib)
        self._check_key('iScansPositively', 0)
        self._check_key('jScansPositively', 1)

    def test_projection_centre(self):
        grid_definition_template_30(self.test_cube, self.mock_grib)
        self._check_key("projectionCentreFlag", 0)

    def test_projection_centre_south_pole(self):
        cs = LambertConformal(central_lat=39.0, central_lon=-96.0,
                              false_easting=0.0, false_northing=0.0,
                              secant_latitudes=(-33, -45),
                              ellipsoid=self.default_ellipsoid)
        test_cube = self._make_test_cube(cs=cs)
        grid_definition_template_30(test_cube, self.mock_grib)
        self._check_key("projectionCentreFlag", 1)


if __name__ == "__main__":
    tests.main()
