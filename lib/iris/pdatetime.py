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
"""
Class for representing partial datetime attributes for sophisticated
constraint handling involving reference time coordinates.

"""

import collections
import datetime
import netcdftime
import operator


_partial_datetime_fields = ['year', 'month', 'day', 'hour', 'minute',
                            'second', 'microsecond', 'tzinfo',
                            'calendar']


class PartialDateTime(collections.namedtuple('PartialDateTime',
                                             _partial_datetime_fields)):

    known_time_implementations = (datetime.datetime, datetime.time,
                                  datetime.date, netcdftime.datetime)

    @property
    def timetuple(self):
        # http://bugs.python.org/issue8005
        # NOTE: It doesn't even matter what this value is, it just triggers
        # some Python internals to allow us to implement our own comparison
        # operators.
        return tuple(self)

    def __new__(cls, year=None, month=None, day=None, hour=None,
                minute=None, second=None, microsecond=None,
                tzinfo=None, calendar=None):
        return super(PartialDateTime, cls).__new__(cls, year, month, day,
                                                   hour, minute, second,
                                                   microsecond, tzinfo,
                                                   calendar)

    def __gt__(self, other):
        return self._compare(operator.gt, other)

    def __lt__(self, other):
        return self._compare(operator.lt, other)

    def __ge__(self, other):
        return self._compare(operator.ge, other)

    def __le__(self, other):
        return self._compare(operator.le, other)

    def __eq__(self, other):
        return self._compare(operator.eq, other)

    def __ne__(self, other):
        return not self == other

    def _compare(self, op, other):
        if not isinstance(other, PartialDateTime.known_time_implementations):
            result = NotImplemented
        elif isinstance(other, PartialDateTime):
            # This is a harder problem, so for now, just don't implement it.
            result = NotImplemented
        else:
            result = True
            for attr_name, attr in zip(self._fields, self):
                if result and attr is not None:
                    if attr_name == 'calendar':
                        calendar = getattr(other, 'calendar', 'gregorian')
                        result = attr == calendar
                        if op is operator.ne:
                            result = not result
                    else:
                        result = op(attr, getattr(other, attr_name))
        return result
