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
Test function :func:`iris.fileformats.grib._load_convert.validity_time_coord.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.coords import DimCoord
from iris.fileformats.grib._load_convert import validity_time_coord
from iris.unit import Unit


class Test(tests.IrisTest):
    def setUp(self):
        self.fp = DimCoord(5, standard_name='forecast_period', units='hours')
        self.fp_test_bounds = np.array([[1.0, 9.0]])
        self.unit = Unit('hours since epoch')
        self.frt = DimCoord(10, standard_name='forecast_reference_time',
                            units=self.unit)

    def test_frt_shape(self):
        frt = mock.Mock(shape=(2,))
        fp = mock.Mock(shape=(1,))
        emsg = 'scalar forecast reference time'
        with self.assertRaisesRegexp(ValueError, emsg):
            validity_time_coord(frt, fp)

    def test_fp_shape(self):
        frt = mock.Mock(shape=(1,))
        fp = mock.Mock(shape=(2,))
        emsg = 'scalar forecast period'
        with self.assertRaisesRegexp(ValueError, emsg):
            validity_time_coord(frt, fp)

    def test(self):
        coord = validity_time_coord(self.frt, self.fp)
        self.assertIsInstance(coord, DimCoord)
        self.assertEqual(coord.standard_name, 'time')
        self.assertEqual(coord.units, self.unit)
        self.assertEqual(coord.shape, (1,))
        point = self.frt.points[0] + self.fp.points[0]
        self.assertEqual(coord.points[0], point)
        self.assertFalse(coord.has_bounds())

    def test_bounded(self):
        self.fp.bounds = self.fp_test_bounds
        coord = validity_time_coord(self.frt, self.fp)
        self.assertIsInstance(coord, DimCoord)
        self.assertEqual(coord.standard_name, 'time')
        self.assertEqual(coord.units, self.unit)
        self.assertEqual(coord.shape, (1,))
        point = self.frt.points[0] + self.fp.points[0]
        self.assertEqual(coord.points[0], point)
        self.assertTrue(coord.has_bounds())
        bounds = self.frt.points[0] + self.fp_test_bounds
        self.assertArrayAlmostEqual(coord.bounds, bounds)


if __name__ == '__main__':
    tests.main()
