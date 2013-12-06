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
import netcdftime

import collections
import datetime
import operator
import warnings


def enhance_datetimes(unit, datetimes):
    """
    Wrap our datetime-like objects, enhancing them with numeric comparison
    capabilities.

    Args:

    * wrapped_object (:class:'datetime.datetime' or \
:class:'netcdftime.datetime'):
        This object or iterable of this object is to be wrapped.
    * unit (:class:'iris.unit.Unit'):
        Unit to be associated with the given datetime-like objects.

    Returns:
        An iterable of wrapped datetype-like object with capability for
        numeric comparison through unit handling.

    .. deprecated:: 1.6

        Datetimes comparison with numeric values is considered deprecated.
        See :class:'iris.partial_datetime.PartialDateTime'.

    """
    if not isinstance(datetimes, collections.Iterable):
        datetimes = [datetimes]
    for ind, dt in enumerate(datetimes):
        datetimes[ind] = DatetimeWrap(dt, unit)
    return datetimes


class DatetimeWrap(object):
    def __init__(self, wrapped_object, unit):
        """
        Wrapper around a datetime-like object giving capability of numeric
        comparison.

        Required only for backwards compatibility with constraining a time
        coord by numeric values rather than partial datetimes.

        Args:

        * wrapped_object (:class:'datetime.datetime' or \
:class:'netcdftime.datetime'):
            This object is the object to be wrapped.
        * unit (:class:'iris.unit.Unit'):
            Unit to be associated with the given datetime-like object.

        Returns:
            A wrapped version of the given datetype-like object with
            capability for numeric comparison through unit handling.

        .. deprecated:: 1.6

            Wrapping datetime-like objects for enhancement is for handling
            datetime comparison with numeric values, which is considered
            deprecated.  See :class:'iris.partial_datetime.PartialDateTime'.

        """
        self.wrapped_object = wrapped_object
        self.unit = unit

    def __getattribute__(self, attr):
        # Defining the getattribute special method in this way avoids infinite
        # recursion.
        try:
            wrapped_object = object.__getattribute__(self, 'wrapped_object')
            return getattr(wrapped_object, attr)
        except AttributeError:
            return object.__getattribute__(self, attr)

    def __str__(self):
        return self.wrapped_object.__str__()

    def __repr__(self):
        return self.wrapped_object.__repr__()

    def __float__(self):
        return self.unit.date2num(self.wrapped_object)

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
        return self._compare(operator.ne, other)

    def _compare(self, op, other):
        if isinstance(other, (int, float)):
            msg = ('Comparing datetime objects with numeric objects (int, '
                   'float) is being deprecated, consider switching to using '
                   'iris.partial_datetime.PartialDateTime objects')
            warnings.warn(msg, DeprecationWarning)
            return op(float(self), other)
        else:
            # Netcdftime does not return NotImplemented
            try:
                return op(self.wrapped_object, other)
            except AttributeError:
                return NotImplemented


class PartialDateTime(collections.namedtuple(
        'PartialDateTime',
        ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'])):
    """
    Object which allows datetime-like object partial comparisons.

    Args:

    * year (int):
        Default is None.
    * month (int):
        Default is None.
    * day (int):
        Default is None
    * hour (int):
        Default is None
    * minute (int):
        Default is None
    * second (int):
        Default is None
    * microsecond (int):
        Default is None

    .. note::

        'calendar' and 'tzinfo' arguments of the datetime object are not
        accepted, as PartialDateTime is currently not calendar aware, i.e. the
        comparison operators do not consider the date but the numeric values
        that represent them.

    For Example:

    >>> from iris.partial_datetime import PartialDateTime
    >>> import datetime
    >>> pdt = PartialDateTime(year=1970)
    >>> dt = datetime.datetime(1970, 1, 1, 6, 0, 0)
    >>> pdt == dt
    True

    """

    known_time_implementations = (datetime.datetime, datetime.time,
                                  datetime.date, netcdftime.datetime,
                                  DatetimeWrap)

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
