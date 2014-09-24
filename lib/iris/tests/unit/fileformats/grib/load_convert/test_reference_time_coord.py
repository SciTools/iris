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
Test function :func:`iris.fileformats.grib._load_convert.reference_time_coord.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from datetime import datetime

from iris.coords import DimCoord
from iris.fileformats.grib._load_convert import reference_time_coord
from iris.unit import CALENDAR_GREGORIAN, Unit


class Test(tests.IrisTest):
    def test(self):
        section = {'year': 2007,
                   'month': 1,
                   'day': 15,
                   'hour': 0,
                   'minute': 3,
                   'second': 0}
        # The call being tested.
        coord = reference_time_coord(section)
        self.assertIsInstance(coord, DimCoord)
        unit = Unit('hours since epoch', calendar=CALENDAR_GREGORIAN)
        self.assertEqual(coord.standard_name, 'forecast_reference_time')
        self.assertEqual(coord.units, unit)
        dt = datetime(section['year'], section['month'], section['day'],
                      section['hour'], section['minute'], section['second'])
        self.assertEqual(coord.shape, (1,))
        self.assertEqual(coord.units.num2date(coord.points[0]), dt)


if __name__ == '__main__':
    tests.main()
