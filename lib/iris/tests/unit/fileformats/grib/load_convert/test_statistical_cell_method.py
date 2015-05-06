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
Tests for function
:func:`iris.fileformats.grib._load_convert.statistical_cell_method`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.exceptions import TranslationError

from iris.fileformats.grib._load_convert import statistical_cell_method


class Test(tests.IrisTest):
    def setUp(self):
        self.section = {}
        self.section['numberOfTimeRange'] = 1
        self.section['typeOfStatisticalProcessing'] = 0
        self.section['typeOfTimeIncrement'] = 2
        self.section['timeIncrement'] = 0

    def test_basic(self):
        cell_method = statistical_cell_method(self.section)
        self.assertEqual(cell_method.method, 'mean')
        self.assertEqual(cell_method.coord_names, ('time',))
        self.assertEqual(cell_method.intervals, ())

    def test_intervals(self):
        self.section['timeIncrement'] = 3
        self.section['indicatorOfUnitForTimeIncrement'] = 1
        cell_method = statistical_cell_method(self.section)
        self.assertEqual(cell_method.method, 'mean')
        self.assertEqual(cell_method.coord_names, ('time',))
        self.assertEqual(cell_method.intervals, ('3 hours',))

    def test_fail_bad_ranges(self):
        self.section['numberOfTimeRange'] = 0
        with self.assertRaisesRegexp(TranslationError,
                                     'aggregation over "0 time ranges"'):
            statistical_cell_method(self.section)

    def test_fail_multiple_ranges(self):
        self.section['numberOfTimeRange'] = 2
        with self.assertRaisesRegexp(TranslationError,
                                     'multiple time ranges \[2\]'):
            statistical_cell_method(self.section)

    def test_fail_unknown_statistic(self):
        self.section['typeOfStatisticalProcessing'] = 17
        with self.assertRaisesRegexp(
                TranslationError,
                'statistical process type \[17\] is not supported'):
            statistical_cell_method(self.section)

    def test_fail_bad_increment_type(self):
        self.section['typeOfTimeIncrement'] = 7
        with self.assertRaisesRegexp(
                TranslationError,
                'time-increment type \[7\] is not supported'):
            statistical_cell_method(self.section)


if __name__ == '__main__':
    tests.main()
