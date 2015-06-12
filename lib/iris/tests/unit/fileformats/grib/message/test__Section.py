# (C) British Crown Copyright 2014 - 2015, Met Office
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
Unit tests for `iris.fileformats.grib._message._Section`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import numpy as np

from iris.fileformats.grib._message import _Section


@tests.skip_data
class Test___getitem__(tests.IrisTest):
    def setUp(self):
        filename = tests.get_data_path(('GRIB', 'uk_t', 'uk_t.grib2'))
        with open(filename, 'rb') as grib_fh:
            self.grib_id = gribapi.grib_new_from_file(grib_fh)

    def test_scalar(self):
        section = _Section(self.grib_id, None, ['Ni'])
        self.assertEqual(section['Ni'], 47)

    def test_array(self):
        section = _Section(self.grib_id, None, ['codedValues'])
        codedValues = section['codedValues']
        self.assertEqual(codedValues.shape, (1551,))
        self.assertArrayAlmostEqual(codedValues[:3],
                                    [-1.78140259, -1.53140259, -1.28140259])

    def test_typeOfFirstFixedSurface(self):
        section = _Section(self.grib_id, None, ['typeOfFirstFixedSurface'])
        self.assertEqual(section['typeOfFirstFixedSurface'], 100)

    def test_numberOfSection(self):
        n = 4
        section = _Section(self.grib_id, n, ['numberOfSection'])
        self.assertEqual(section['numberOfSection'], n)

    def test_invalid(self):
        section = _Section(self.grib_id, None, ['Ni'])
        with self.assertRaisesRegexp(KeyError, 'Nii'):
            section['Nii']


@tests.skip_data
class Test__getitem___pdt_31(tests.IrisTest):
    def setUp(self):
        filename = tests.get_data_path(('GRIB', 'umukv', 'ukv_chan9.grib2'))
        with open(filename, 'rb') as grib_fh:
            self.grib_id = gribapi.grib_new_from_file(grib_fh)
        self.keys = ['satelliteSeries', 'satelliteNumber', 'instrumentType',
                     'scaleFactorOfCentralWaveNumber',
                     'scaledValueOfCentralWaveNumber']

    def test_array(self):
        section = _Section(self.grib_id, None, self.keys)
        for key in self.keys:
            value = section[key]
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.shape, (1,))


if __name__ == '__main__':
    tests.main()
