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
Time handling.

"""
import collections
import datetime
import functools
import operator

import netcdftime


@functools.total_ordering
class PartialDateTime(object):
    """
    A :class:`PartialDateTime` object specifies values for some subset of
    the calendar/time fields (year, month, hour, etc.) for comparing
    with :class:`datetime.datetime`-like instances.

    Comparisons are defined against any other class with all of the
    attributes: year, month, day, hour, minute, second, microsecond.
    Notably, this includes :class:`datetime.datetime` and
    :class:`netcdftime.datetime`.

    A :class:`PartialDateTime` object is not limited to any particular
    calendar, so no restriction is placed on the range of values
    allowed in its component fields. Thus, it is perfectly legitimate to
    create an instance as: `PartialDateTime(month=2, day=30)`.

    """

    __slots__ = ('year', 'month', 'day', 'hour', 'minute', 'second',
                 'microsecond')

    #: A dummy value provided as a workaround to allow comparisons with
    #: :class:`datetime.datetime`.
    #: See http://bugs.python.org/issue8005.
    # NB. It doesn't even matter what this value is.
    timetuple = None

    def __init__(self, year=None, month=None, day=None, hour=None,
                 minute=None, second=None, microsecond=None):
        """
        Allows partial comparisons against datetime-like objects.

        Args:

        * year (int):
        * month (int):
        * day (int):
        * hour (int):
        * minute (int):
        * second (int):
        * microsecond (int):

        For example, to select any days of the year after the 3rd of April:

        >>> from iris.time import PartialDateTime
        >>> import datetime
        >>> pdt = PartialDateTime(month=4, day=3)
        >>> datetime.datetime(2014, 4, 1) > pdt
        False
        >>> datetime.datetime(2014, 4, 5) > pdt
        True
        >>> datetime.datetime(2014, 5, 1) > pdt
        True
        >>> datetime.datetime(2015, 2, 1) > pdt
        False

        """

        #: The year number as an integer, or None.
        self.year = year
        #: The month number as an integer, or None.
        self.month = month
        #: The day number as an integer, or None.
        self.day = day
        #: The hour number as an integer, or None.
        self.hour = hour
        #: The minute number as an integer, or None.
        self.minute = minute
        #: The second number as an integer, or None.
        self.second = second
        #: The microsecond number as an integer, or None.
        self.microsecond = microsecond

    def __gt__(self, other):
        if isinstance(other, type(self)):
            raise TypeError('Cannot order PartialDateTime instances.')
        result = False
        try:
            for attr_name in self.__slots__:
                attr = getattr(self, attr_name)
                other_attr = getattr(other, attr_name)
                if attr is not None and attr != other_attr:
                    result = attr > other_attr
                    break
        except AttributeError:
            result = NotImplemented
        return result

    def __eq__(self, other):
        if isinstance(other, type(self)):
            slots = self.__slots__
            self_tuple = tuple(getattr(self, name) for name in slots)
            other_tuple = tuple(getattr(other, name) for name in slots)
            result = self_tuple == other_tuple
        else:
            result = True
            try:
                for attr_name in self.__slots__:
                    attr = getattr(self, attr_name)
                    other_attr = getattr(other, attr_name)
                    if attr is not None and attr != other_attr:
                        result = False
                        break
            except AttributeError:
                result = NotImplemented
        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def __cmp__(self, other):
        # Since we've defined all the rich comparison operators (via
        # functools.total_ordering), we can only reach this point if
        # neither this class nor the other class had a rich comparison
        # that could handle the type combination.
        # We don't want Python to fall back to the default `object`
        # behaviour (which compares using object IDs), so we raise an
        # exception here instead.
        fmt = 'unable to compare PartialDateTime with {}'
        raise TypeError(fmt.format(type(other)))
