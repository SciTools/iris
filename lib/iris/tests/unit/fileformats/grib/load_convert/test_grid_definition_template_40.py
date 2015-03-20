# (C) British Crown Copyright 2015, Met Office
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
:func:`iris.fileformats.grib._load_convert.grid_definition_template_40`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import iris.coord_systems
import iris.coords
from iris.tests.unit.fileformats.grib.load_convert import empty_metadata
from iris.fileformats.grib._load_convert import grid_definition_template_40


MDI = 2 ** 32 - 1


class Test_regular(tests.IrisTest):

    def section_3(self):
        section = {
            'shapeOfTheEarth': 0,
            'scaleFactorOfRadiusOfSphericalEarth': 0,
            'scaledValueOfRadiusOfSphericalEarth': 6367470,
            'scaleFactorOfEarthMajorAxis': 0,
            'scaledValueOfEarthMajorAxis': MDI,
            'scaleFactorOfEarthMinorAxis': 0,
            'scaledValueOfEarthMinorAxis': MDI,
            'iDirectionIncrement': 22500000,
            'longitudeOfFirstGridPoint': 0,
            'Ni': 16,
            'scanningMode': 0b01000000,
            'distinctLatitudes': np.array([-73.79921363, -52.81294319,
                                           -31.70409175, -10.56988231,
                                           10.56988231,  31.70409175,
                                           52.81294319,  73.79921363]),
            'numberOfOctectsForNumberOfPoints': 0,
            'interpretationOfNumberOfPoints': 0,
        }
        return section

    def expected(self, y_dim, x_dim, y_neg=True):
        # Prepare the expectation.
        expected = empty_metadata()
        cs = iris.coord_systems.GeogCS(6367470)
        nx = 16
        dx = 22.5
        x_origin = 0
        x = iris.coords.DimCoord(np.arange(nx) * dx + x_origin,
                                 standard_name='longitude',
                                 units='degrees_east',
                                 coord_system=cs,
                                 circular=True)
        y_points = np.array([73.79921363, 52.81294319,
                             31.70409175, 10.56988231,
                             -10.56988231,  -31.70409175,
                             -52.81294319,  -73.79921363])
        if not y_neg:
            y_points = y_points[::-1]
        y = iris.coords.DimCoord(y_points,
                                 standard_name='latitude',
                                 units='degrees_north',
                                 coord_system=cs)
        expected['dim_coords_and_dims'].append((y, y_dim))
        expected['dim_coords_and_dims'].append((x, x_dim))
        return expected

    def test(self):
        section = self.section_3()
        metadata = empty_metadata()
        grid_definition_template_40(section, metadata)
        expected = self.expected(0, 1, y_neg=False)
        self.assertEqual(metadata, expected)

    def test_transposed(self):
        section = self.section_3()
        section['scanningMode'] = 0b01100000
        metadata = empty_metadata()
        grid_definition_template_40(section, metadata)
        expected = self.expected(1, 0, y_neg=False)
        self.assertEqual(metadata, expected)

    def test_reverse_latitude(self):
        section = self.section_3()
        section['scanningMode'] = 0b00000000
        metadata = empty_metadata()
        grid_definition_template_40(section, metadata)
        expected = self.expected(0, 1, y_neg=True)
        self.assertEqual(metadata, expected)


class Test_reduced(tests.IrisTest):

    def section_3(self):
        section = {
            'shapeOfTheEarth': 0,
            'scaleFactorOfRadiusOfSphericalEarth': 0,
            'scaledValueOfRadiusOfSphericalEarth': 6367470,
            'scaleFactorOfEarthMajorAxis': 0,
            'scaledValueOfEarthMajorAxis': MDI,
            'scaleFactorOfEarthMinorAxis': 0,
            'scaledValueOfEarthMinorAxis': MDI,
            'longitudes': np.array([0., 180.,
                                    0., 120., 240.,
                                    0., 120., 240.,
                                    0., 180.]),
            'latitudes': np.array([-59.44440829, -59.44440829,
                                   -19.87571915, -19.87571915, -19.87571915,
                                   19.87571915, 19.87571915, 19.87571915,
                                   59.44440829, 59.44440829]),
            'numberOfOctectsForNumberOfPoints': 1,
            'interpretationOfNumberOfPoints': 1,
        }
        return section

    def expected(self):
        # Prepare the expectation.
        expected = empty_metadata()
        cs = iris.coord_systems.GeogCS(6367470)
        x_points = np.array([0., 180.,
                             0., 120., 240.,
                             0., 120., 240.,
                             0., 180.])
        y_points = np.array([-59.44440829, -59.44440829,
                             -19.87571915, -19.87571915, -19.87571915,
                             19.87571915, 19.87571915, 19.87571915,
                             59.44440829, 59.44440829])
        x = iris.coords.AuxCoord(x_points,
                                 standard_name='longitude',
                                 units='degrees_east',
                                 coord_system=cs)
        y = iris.coords.AuxCoord(y_points,
                                 standard_name='latitude',
                                 units='degrees_north',
                                 coord_system=cs)
        expected['aux_coords_and_dims'].append((y, 0))
        expected['aux_coords_and_dims'].append((x, 0))
        return expected

    def test(self):
        section = self.section_3()
        metadata = empty_metadata()
        expected = self.expected()
        grid_definition_template_40(section, metadata)
        self.assertEqual(metadata, expected)


if __name__ == '__main__':
    tests.main()
