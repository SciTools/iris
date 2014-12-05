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
Test function :func:`iris.fileformats.grib._load_convert.other_time_coord.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import iris.coords
from iris.fileformats.grib._load_convert import other_time_coord


class TestValid(tests.IrisTest):
    def test_t(self):
        rt = iris.coords.DimCoord(48, 'time', units='hours since epoch')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        result = other_time_coord(rt, fp)
        expected = iris.coords.DimCoord(42, 'forecast_reference_time',
                                        units='hours since epoch')
        self.assertEqual(result, expected)

    def test_frt(self):
        rt = iris.coords.DimCoord(48, 'forecast_reference_time',
                                  units='hours since epoch')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        result = other_time_coord(rt, fp)
        expected = iris.coords.DimCoord(54, 'time', units='hours since epoch')
        self.assertEqual(result, expected)


class TestInvalid(tests.IrisTest):
    def test_t_with_bounds(self):
        rt = iris.coords.DimCoord(48, 'time', units='hours since epoch',
                                  bounds=[36, 60])
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'bounds'):
            other_time_coord(rt, fp)

    def test_frt_with_bounds(self):
        rt = iris.coords.DimCoord(48, 'forecast_reference_time',
                                  units='hours since epoch',
                                  bounds=[42, 54])
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'bounds'):
            other_time_coord(rt, fp)

    def test_fp_with_bounds(self):
        rt = iris.coords.DimCoord(48, 'time', units='hours since epoch')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours',
                                  bounds=[3, 9])
        with self.assertRaisesRegexp(ValueError, 'bounds'):
            other_time_coord(rt, fp)

    def test_vector_t(self):
        rt = iris.coords.DimCoord([0, 3], 'time', units='hours since epoch')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'Vector'):
            other_time_coord(rt, fp)

    def test_vector_frt(self):
        rt = iris.coords.DimCoord([0, 3], 'forecast_reference_time',
                                  units='hours since epoch')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'Vector'):
            other_time_coord(rt, fp)

    def test_vector_fp(self):
        rt = iris.coords.DimCoord(48, 'time', units='hours since epoch')
        fp = iris.coords.DimCoord([6, 12], 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'Vector'):
            other_time_coord(rt, fp)

    def test_invalid_rt_name(self):
        rt = iris.coords.DimCoord(1, 'height')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'reference time'):
            other_time_coord(rt, fp)

    def test_invalid_t_unit(self):
        rt = iris.coords.DimCoord(1, 'time', units='Pa')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'unit.*Pa'):
            other_time_coord(rt, fp)

    def test_invalid_frt_unit(self):
        rt = iris.coords.DimCoord(1, 'forecast_reference_time', units='km')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='hours')
        with self.assertRaisesRegexp(ValueError, 'unit.*km'):
            other_time_coord(rt, fp)

    def test_invalid_fp_unit(self):
        rt = iris.coords.DimCoord(48, 'forecast_reference_time',
                                  units='hours since epoch')
        fp = iris.coords.DimCoord(6, 'forecast_period', units='kg')
        with self.assertRaisesRegexp(ValueError, 'unit.*kg'):
            other_time_coord(rt, fp)


if __name__ == '__main__':
    tests.main()
