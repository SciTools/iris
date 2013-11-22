# (C) British Crown Copyright 2013, Met Office
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
"""Experimental constraints, interface liable to change."""

import datetime
import netcdftime

from iris._constraints import *
from iris._constraints import _ColumnIndexManager
from iris.util import _fix_netcdftime_datetime


class TimeRangeConstraint(Constraint):
    """
    Range selection from a time coordinate.

    Currently, only day-of-year is supported.

    """
    def __init__(self, day_of_year=None, coord='time'):
        """
        Creates a TimeRangeConstraint.

        kwargs:

            * day_of_year - Tuple of (start, end) defining an inclusive range.
                Both start and end are a (month, day) tuple.

            * coord - The coord to constrain, defaulting to 'time'.

        Example::

            nov_dec = TimeRangeConstraint(day_of_year=((11, 01), (12,31))

        """
        self.day_of_year, self.coord = day_of_year, coord
        Constraint.__init__(self)

    def __repr__(self):
        return 'TimeRangeConstraint(day_of_year={}, coord={})'.format(
            self.day_of_year, self.coord)

    def _day_of_year(self, dt):
        """
        Convenience method to extract the day of the year from either a
        :class:`datetime.datetime` or a :class:`netcdftime.datetime`.

        Caution: See also :func:`iris.util._fix_netcdftime_datetime`.

        """
        # netcdf.datetime?
        if hasattr(dt, 'dayofyr'):
            result = dt.dayofyr
        # datetime.datetime
        else:
            result = dt.timetuple().tm_yday
        return result

    def _contains_point(self, point, unit):
        point_dt = unit.num2date(point)
        point_yday = self._day_of_year(point_dt)

        start_dt = netcdftime.datetime(point_dt.year, *self.day_of_year[0])
        start_dt = _fix_netcdftime_datetime(start_dt, unit)
        start_yday = self._day_of_year(start_dt)

        end_dt = netcdftime.datetime(point_dt.year, *self.day_of_year[1])
        end_dt = _fix_netcdftime_datetime(end_dt, unit)
        end_yday = self._day_of_year(end_dt)

        if start_yday < end_yday:
            result = start_yday <= point_yday <= end_yday
        elif start_yday > end_yday:
            result = point_yday >= start_yday or point_yday <= end_yday
        else:
            raise ValueError('start_yday == end_yday')
        return result

    def _CIM_extract(self, cube):
        result_cim = _ColumnIndexManager(len(cube.shape))
        try:
            coord = cube.coord(self.coord)
        except iris.exceptions.CoordinateNotFoundError:
            coord = None

        if coord:
            dims = cube.coord_dims(coord)
            if len(dims) > 1:
                raise iris.exceptions.CoordinateMultiDimError(coord)

            # Vector coord?
            if dims:
                inout = [self._contains_point(point, coord.units)
                         for point in coord.points]
                result_cim[dims[0]] = np.array(inout)

            # Scalar coord.
            else:
                if not self._contains_point(coord.points[0], coord.units):
                    result_cim.all_false()
        else:
            result_cim.all_false()

        return result_cim
