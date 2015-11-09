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
Test function :func:`iris.fileformats.grib._load_convert.time_range_unit.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import time_range_unit
from cf_units import Unit


class Test(tests.IrisTest):
    def setUp(self):
        self.unit_by_indicator = {0: Unit('minutes'),
                                  1: Unit('hours'),
                                  2: Unit('days'),
                                  10: Unit('3 hours'),
                                  11: Unit('6 hours'),
                                  12: Unit('12 hours'),
                                  13: Unit('seconds')}

    def test_units(self):
        for indicator, unit in self.unit_by_indicator.items():
            result = time_range_unit(indicator)
            self.assertEqual(result, unit)

    def test_bad_indicator(self):
        emsg = 'unsupported time range'
        with self.assertRaisesRegexp(TranslationError, emsg):
            time_range_unit(-1)


if __name__ == '__main__':
    tests.main()
