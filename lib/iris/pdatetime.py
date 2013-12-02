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
                            'second', 'microsecond']
# NOTE: 'calendar' and 'tzinfo' is not included as PartialDateTime is not
# calendar aware, i.e. the comparison operators do not consider the date
# values and adjustment by tzinfo or calendar.


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
                minute=None, second=None, microsecond=None):
        return super(PartialDateTime, cls).__new__(cls, year, month, day,
                                                   hour, minute, second,
                                                   microsecond)

    def __gt__(self, other):
        result = self._compare(other)
        if result is False:
            result = self._comparison_x(operator.gt, other)
        return result

    def __lt__(self, other):
        result = self._compare(other)
        if result is False:
            result = self._comparison_x(operator.lt, other)
        return result

    def __ge__(self, other):
        result = self._compare(other)
        if result is False:
            result = self._comparison_xe(operator.ge, other)
        return result

    def __le__(self, other):
        result = self._compare(other)
        if result is False:
            result = self._comparison_xe(operator.le, other)
        return result

    def __eq__(self, other):
        result = self._compare(other)
        if result is False:
            result = self._comparison_eq(operator.eq, other)
        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def _comparison_x(self, op, other):
        # Handling of comparison operators that stop on first anything other
        # than a 'None', which represents the case where 'a == b' (i.e. skip
        # this field).
        result = False
        for attr_name, attr in zip(self._fields, self):
            if attr is not None:
                result = op(attr, getattr(other, attr_name))
                if result is False:
                    if attr == getattr(other, attr_name):
                        result = None
                if result in (True, False, NotImplemented):
                    break
        if result is None:
            result = False
        return result

    def _comparison_xe(self, op, other):
        # Handling of comparison operators that stop on first True.
        result = False
        for attr_name, attr in zip(self._fields, self):
            if attr is not None:
                result = op(attr, getattr(other, attr_name))
                if result is True:
                    if attr == getattr(other, attr_name):
                        result = None
                if result in (True, False, NotImplemented):
                    break
        if result is None:
            result = True
        return result

    def _comparison_eq(self, op, other):
        # Handling of comparison operators that stop on first True.
        result = False
        for attr_name, attr in zip(self._fields, self):
            if attr is not None:
                result = op(attr, getattr(other, attr_name))
                if result in (False, NotImplemented):
                    break
        return result

    def _compare(self, other):
        if not isinstance(other, PartialDateTime.known_time_implementations):
            result = NotImplemented
        elif isinstance(other, PartialDateTime):
            # This is a harder problem, so for now, just don't implement it.
            result = NotImplemented
        else:
            result = False
        return result
