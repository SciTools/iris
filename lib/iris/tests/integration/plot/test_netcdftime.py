# (C) British Crown Copyright 2016 - 2018, Met Office
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
Test plot of time coord with non-gregorian calendar.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import cftime
import numpy as np

from iris.coords import AuxCoord

from cf_units import Unit
if tests.NC_TIME_AXIS_AVAILABLE:
    from nc_time_axis import CalendarDateTime


# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt
    import iris.plot as iplt


@tests.skip_nc_time_axis
@tests.skip_plot
class Test(tests.GraphicsTest):
    def test_360_day_calendar(self):
        n = 360
        calendar = '360_day'
        time_unit = Unit('days since 1970-01-01 00:00', calendar=calendar)
        time_coord = AuxCoord(np.arange(n), 'time', units=time_unit)
        times = [time_unit.num2date(point) for point in time_coord.points]
        times = [cftime.datetime(atime.year, atime.month, atime.day,
                                 atime.hour, atime.minute, atime.second)
                 for atime in times]
        expected_ydata = np.array([CalendarDateTime(time, calendar)
                                   for time in times])
        line1, = iplt.plot(time_coord)
        result_ydata = line1.get_ydata()
        self.assertArrayEqual(expected_ydata, result_ydata)


if __name__ == "__main__":
    tests.main()
