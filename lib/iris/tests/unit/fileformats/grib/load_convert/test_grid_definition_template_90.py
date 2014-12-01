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
:func:`iris.fileformats.grib._load_convert.grid_definition_template_90`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from collections import OrderedDict

import numpy as np

import iris.coord_systems
import iris.coords
import iris.exceptions
from iris.fileformats.grib._load_convert import grid_definition_template_90


MDI = 2 ** 32 - 1


class Test(tests.IrisTest):
    def empty_metadata(self):
        metadata = OrderedDict()
        metadata['factories'] = []
        metadata['references'] = []
        metadata['standard_name'] = None
        metadata['long_name'] = None
        metadata['units'] = None
        metadata['attributes'] = {}
        metadata['cell_methods'] = []
        metadata['dim_coords_and_dims'] = []
        metadata['aux_coords_and_dims'] = []
        return metadata

    def uk(self):
        section = {
            'shapeOfTheEarth': 3,
            'scaleFactorOfRadiusOfSphericalEarth': MDI,
            'scaledValueOfRadiusOfSphericalEarth': MDI,
            'scaleFactorOfEarthMajorAxis': 4,
            'scaledValueOfEarthMajorAxis': 63781688,
            'scaleFactorOfEarthMinorAxis': 4,
            'scaledValueOfEarthMinorAxis': 63565840,
            'Nx': 390,
            'Ny': 227,
            'latitudeOfSubSatellitePoint': 0,
            'longitudeOfSubSatellitePoint': 0,
            'resolutionAndComponentFlags': 0,
            'dx': 3622,
            'dy': 3610,
            'Xp': 1856000,
            'Yp': 1856000,
            'scanningMode': 192,
            'orientationOfTheGrid': 0,
            'Nr': 6610674,
            'Xo': 1733,
            'Yo': 3320
        }
        return section

    def expected_uk(self, y_dim, x_dim):
        # Prepare the expectation.
        expected = self.empty_metadata()
        major = 6378168.8
        ellipsoid = iris.coord_systems.GeogCS(major, 6356584.0)
        height = (6610674e-6 - 1) * major
        lat = lon = 0
        easting = northing = 0
        cs = iris.coord_systems.VerticalPerspective(lat, lon, height, easting,
                                                    northing, ellipsoid)
        nx = 390
        x_origin = 369081.56145444815
        dx = -3000.663101255676
        x = iris.coords.DimCoord(np.arange(nx) * dx + x_origin,
                                 'projection_x_coordinate', units='m',
                                 coord_system=cs)
        ny = 227
        y_origin = 4392884.59201891
        dy = 3000.604229521113
        y = iris.coords.DimCoord(np.arange(ny) * dy + y_origin,
                                 'projection_y_coordinate', units='m',
                                 coord_system=cs)
        expected['dim_coords_and_dims'].append((y, y_dim))
        expected['dim_coords_and_dims'].append((x, x_dim))
        return expected

    def compare(self, metadata, expected):
        # Compare the result with the expectation.
        self.assertEqual(len(metadata['dim_coords_and_dims']),
                         len(expected['dim_coords_and_dims']))
        for result_pair, expected_pair in zip(metadata['dim_coords_and_dims'],
                                              expected['dim_coords_and_dims']):
            result_coord, result_dims = result_pair
            expected_coord, expected_dims = expected_pair
            # Ensure the dims match.
            self.assertEqual(result_dims, expected_dims)
            # Ensure the coordinate systems match (allowing for precision).
            result_cs = result_coord.coord_system
            expected_cs = expected_coord.coord_system
            self.assertEqual(type(result_cs), type(expected_cs))
            self.assertEqual(result_cs.latitude_of_projection_origin,
                             expected_cs.latitude_of_projection_origin)
            self.assertEqual(result_cs.longitude_of_projection_origin,
                             expected_cs.longitude_of_projection_origin)
            self.assertAlmostEqual(result_cs.perspective_point_height,
                                   expected_cs.perspective_point_height)
            self.assertEqual(result_cs.false_easting,
                             expected_cs.false_easting)
            self.assertEqual(result_cs.false_northing,
                             expected_cs.false_northing)
            self.assertAlmostEqual(result_cs.ellipsoid.semi_major_axis,
                                   expected_cs.ellipsoid.semi_major_axis)
            self.assertEqual(result_cs.ellipsoid.semi_minor_axis,
                             expected_cs.ellipsoid.semi_minor_axis)
            # Now we can ignore the coordinate systems and compare the
            # rest of the coordinate attributes.
            result_coord.coord_system = None
            expected_coord.coord_system = None
            self.assertEqual(result_coord, expected_coord)

        # Ensure no other metadata was created.
        for name in expected.iterkeys():
            if name == 'dim_coords_and_dims':
                continue
            self.assertEqual(metadata[name], expected[name])

    def test_uk(self):
        section = self.uk()
        metadata = self.empty_metadata()
        grid_definition_template_90(section, metadata)
        expected = self.expected_uk(0, 1)
        self.compare(metadata, expected)

    def test_uk_transposed(self):
        section = self.uk()
        section['scanningMode'] = 0b11100000
        metadata = self.empty_metadata()
        grid_definition_template_90(section, metadata)
        expected = self.expected_uk(1, 0)
        self.compare(metadata, expected)

    def test_non_zero_latitude(self):
        section = self.uk()
        section['latitudeOfSubSatellitePoint'] = 1
        metadata = self.empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError,
                                     'non-zero latitude'):
            grid_definition_template_90(section, metadata)

    def test_rotated_meridian(self):
        section = self.uk()
        section['orientationOfTheGrid'] = 1
        metadata = self.empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError,
                                     'orientation'):
            grid_definition_template_90(section, metadata)

    def test_zero_height(self):
        section = self.uk()
        section['Nr'] = 0
        metadata = self.empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError,
                                     'zero'):
            grid_definition_template_90(section, metadata)

    def test_orthographic(self):
        section = self.uk()
        section['Nr'] = MDI
        metadata = self.empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError,
                                     'orthographic'):
            grid_definition_template_90(section, metadata)

    def test_scanning_mode_positive_x(self):
        section = self.uk()
        section['scanningMode'] = 0b01000000
        metadata = self.empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError, r'\+x'):
            grid_definition_template_90(section, metadata)

    def test_scanning_mode_negative_y(self):
        section = self.uk()
        section['scanningMode'] = 0b10000000
        metadata = self.empty_metadata()
        with self.assertRaisesRegexp(iris.exceptions.TranslationError, '-y'):
            grid_definition_template_90(section, metadata)


if __name__ == '__main__':
    tests.main()
