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
:func:`iris.fileformats.grib._load_convert.grid_definition_template_12`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import iris.coord_systems
import iris.coords
import iris.exceptions
from iris.tests.unit.fileformats.grib.load_convert import empty_metadata
from iris.fileformats.grib._load_convert import grid_definition_template_12


MDI = 2 ** 32 - 1


class Test(tests.IrisTest):
    def section_3(self):
        section = {
            'shapeOfTheEarth': 7,
            'scaleFactorOfRadiusOfSphericalEarth': MDI,
            'scaledValueOfRadiusOfSphericalEarth': MDI,
            'scaleFactorOfEarthMajorAxis': 3,
            'scaledValueOfEarthMajorAxis': 6377563396,
            'scaleFactorOfEarthMinorAxis': 3,
            'scaledValueOfEarthMinorAxis': 6356256909,
            'Ni': 4,
            'Nj': 3,
            'latitudeOfReferencePoint': 49000000,
            'longitudeOfReferencePoint': -2000000,
            'resolutionAndComponentFlags': 0,
            'scaleFactorAtReferencePoint': 0.9996012717,
            'XR': 40000000,
            'YR': -10000000,
            'scanningMode': 64,
            'Di': 200000,
            'Dj': 100000,
            'X1': 29300000,
            'Y1': 9200000,
            'X2': 29900000,
            'Y2': 9400000
        }
        return section

    def expected(self, y_dim, x_dim):
        # Prepare the expectation.
        expected = empty_metadata()
        ellipsoid = iris.coord_systems.GeogCS(6377563.396, 6356256.909)
        cs = iris.coord_systems.TransverseMercator(49, -2, 400000, -100000,
                                                   0.9996012717, ellipsoid)
        nx = 4
        x_origin = 293000
        dx = 2000
        x = iris.coords.DimCoord(np.arange(nx) * dx + x_origin,
                                 'projection_x_coordinate', units='m',
                                 coord_system=cs)
        ny = 3
        y_origin = 92000
        dy = 1000
        y = iris.coords.DimCoord(np.arange(ny) * dy + y_origin,
                                 'projection_y_coordinate', units='m',
                                 coord_system=cs)
        expected['dim_coords_and_dims'].append((y, y_dim))
        expected['dim_coords_and_dims'].append((x, x_dim))
        return expected

    def test(self):
        section = self.section_3()
        metadata = empty_metadata()
        grid_definition_template_12(section, metadata)
        expected = self.expected(0, 1)
        self.assertEqual(metadata, expected)

    def test_spherical(self):
        section = self.section_3()
        section['shapeOfTheEarth'] = 0
        metadata = empty_metadata()
        grid_definition_template_12(section, metadata)
        expected = self.expected(0, 1)
        cs = expected['dim_coords_and_dims'][0][0].coord_system
        cs.ellipsoid = iris.coord_systems.GeogCS(6367470)
        self.assertEqual(metadata, expected)

    def test_negative_x(self):
        section = self.section_3()
        section['scanningMode'] = 0b11000000
        metadata = empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError,
                                     '-x scanning'):
            grid_definition_template_12(section, metadata)

    def test_negative_y(self):
        section = self.section_3()
        section['scanningMode'] = 0b00000000
        metadata = empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError,
                                     '-y scanning'):
            grid_definition_template_12(section, metadata)

    def test_transposed(self):
        section = self.section_3()
        section['scanningMode'] = 0b01100000
        metadata = empty_metadata()
        grid_definition_template_12(section, metadata)
        expected = self.expected(1, 0)
        self.assertEqual(metadata, expected)

    def test_di_tolerance(self):
        # Even though Ni * Di doesn't exactly match X1 to X2 it should
        # be close enough to allow the translation.
        section = self.section_3()
        section['X2'] += 1
        metadata = empty_metadata()
        grid_definition_template_12(section, metadata)
        expected = self.expected(0, 1)
        x = expected['dim_coords_and_dims'][1][0]
        x.points = np.linspace(293000, 299000.01, 4)
        self.assertEqual(metadata, expected)

    def test_incompatible_grid_extent(self):
        section = self.section_3()
        section['X2'] += 100
        metadata = empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError,
                                     'grid'):
            grid_definition_template_12(section, metadata)

    def test_scale_workaround(self):
        section = self.section_3()
        section['scaleFactorAtReferencePoint'] = 1065346526
        metadata = empty_metadata()
        grid_definition_template_12(section, metadata)
        expected = self.expected(0, 1)
        # A float32 can't hold exactly the same value.
        cs = expected['dim_coords_and_dims'][0][0].coord_system
        cs.scale_factor_at_central_meridian = 0.9996012449264526
        self.assertEqual(metadata, expected)


if __name__ == '__main__':
    tests.main()
