# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test plot of time coord with non-standard calendar.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from cf_units import Unit
import cftime
import numpy as np

from iris.coords import AuxCoord

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_nc_time_axis
@tests.skip_plot
class Test(tests.GraphicsTest):
    def test_360_day_calendar(self):
        n = 360
        calendar = "360_day"
        time_unit = Unit("days since 1970-01-01 00:00", calendar=calendar)
        time_coord = AuxCoord(np.arange(n), "time", units=time_unit)
        times = [time_unit.num2date(point) for point in time_coord.points]
        times = [
            cftime.datetime(
                atime.year,
                atime.month,
                atime.day,
                atime.hour,
                atime.minute,
                atime.second,
                calendar=calendar,
            )
            for atime in times
        ]

        expected_ydata = times
        (line1,) = iplt.plot(time_coord)
        result_ydata = line1.get_ydata()
        self.assertArrayEqual(expected_ydata, result_ydata)


if __name__ == "__main__":
    tests.main()
