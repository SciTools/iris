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
:func:`iris.fileformats.grib._load_convert.grid_definition_template_50`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import iris.coords
from iris.tests.unit.fileformats.grib.load_convert import empty_metadata
from iris.fileformats.grib._load_convert import grid_definition_template_50


MDI = 2 ** 32 - 1


class Test(tests.IrisTest):

    params = {'triangular': {'J': 5, 'K': 5, 'M': 5},
              'rhomboidal': {'J': 5, 'K': 10, 'M': 5},
              'trapezoidal': {'J': 5, 'K': 5, 'M': 3}}

    def section_3(self, truncation_type):
        section = {
            'shapeOfTheEarth': 0,
            'scaleFactorOfRadiusOfSphericalEarth': 0,
            'scaledValueOfRadiusOfSphericalEarth': 6367470,
            'scaleFactorOfEarthMajorAxis': 0,
            'scaledValueOfEarthMajorAxis': MDI,
            'scaleFactorOfEarthMinorAxis': 0,
            'scaledValueOfEarthMinorAxis': MDI,
            'J': self.params[truncation_type]['J'],
            'K': self.params[truncation_type]['K'],
            'M': self.params[truncation_type]['M'],
        }
        return section

    def expected(self, truncation_type):
        expected = empty_metadata()
        if truncation_type == 'triangular':
            m_points = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                        2, 2, 2, 2, 3, 3, 3, 4, 4, 5]
            n_points = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                        2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
        elif truncation_type == 'rhomboidal':
            m_points = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                        3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
            n_points = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7,
                        3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 10]
        elif truncation_type == 'trapezoidal':
            m_points = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
            n_points = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5]
        m_coord = iris.coords.AuxCoord(m_points, long_name='zonal_wavenumber',
                                       units=1, coord_system=None)
        n_coord = iris.coords.AuxCoord(n_points,
                                       long_name='meridional_wavenumber',
                                       units=1, coord_system=None)
        expected['aux_coords_and_dims'].append((m_coord, 0))
        expected['aux_coords_and_dims'].append((n_coord, 0))
        expected['attributes']['truncation_type'] = truncation_type
        expected['attributes']['J'] = self.params[truncation_type]['J']
        expected['attributes']['K'] = self.params[truncation_type]['K']
        expected['attributes']['M'] = self.params[truncation_type]['M']
        return expected

    def test_triangular(self):
        section = self.section_3('triangular')
        metadata = empty_metadata()
        grid_definition_template_50(section, metadata)
        expected = self.expected('triangular')
        self.assertEqual(metadata, expected)

    def test_rhomboidal(self):
        section = self.section_3('rhomboidal')
        metadata = empty_metadata()
        grid_definition_template_50(section, metadata)
        expected = self.expected('rhomboidal')
        self.assertEqual(metadata, expected)

    def test_trapezoidal(self):
        section = self.section_3('trapezoidal')
        metadata = empty_metadata()
        grid_definition_template_50(section, metadata)
        expected = self.expected('trapezoidal')
        self.assertEqual(metadata, expected)


if __name__ == '__main__':
    tests.main()
