# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Time handling.

"""

import functools


@functools.total_ordering
class PartialDateTime:
    """
    A :class:`PartialDateTime` object specifies values for some subset of
    the calendar/time fields (year, month, hour, etc.) for comparing
    with :class:`datetime.datetime`-like instances.

    Comparisons are defined against any other class with all of the
    attributes: year, month, day, hour, minute, and second.
    Notably, this includes :class:`datetime.datetime` and
    :class:`cftime.datetime`. Comparison also extends to the
    microsecond attribute for classes, such as
    :class:`datetime.datetime`, which define it.

    A :class:`PartialDateTime` object is not limited to any particular
    calendar, so no restriction is placed on the range of values
    allowed in its component fields. Thus, it is perfectly legitimate to
    create an instance as: `PartialDateTime(month=2, day=30)`.

    """

    __slots__ = (
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
    )

    #: A dummy value provided as a workaround to allow comparisons with
    #: :class:`datetime.datetime`.
    #: See http://bugs.python.org/issue8005.
    # NB. It doesn't even matter what this value is.
    timetuple = None

    def __init__(
        self,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
    ):
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

    def __repr__(self):
        attr_pieces = [
            "{}={}".format(name, getattr(self, name))
            for name in self.__slots__
            if getattr(self, name) is not None
        ]
        result = "{}({})".format(type(self).__name__, ", ".join(attr_pieces))
        return result

    def __gt__(self, other):
        if isinstance(other, type(self)):
            raise TypeError("Cannot order PartialDateTime instances.")
        result = False
        try:
            # Everything except 'microsecond' is mandatory
            for attr_name in self.__slots__[:-1]:
                attr = getattr(self, attr_name)
                other_attr = getattr(other, attr_name)
                if attr is not None and attr != other_attr:
                    result = attr > other_attr
                    break
            # 'microsecond' is optional
            if result and hasattr(other, "microsecond"):
                attr = self.microsecond
                other_attr = other.microsecond
                if attr is not None and attr != other_attr:
                    result = attr > other_attr
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
                # Everything except 'microsecond' is mandatory
                for attr_name in self.__slots__[:-1]:
                    attr = getattr(self, attr_name)
                    other_attr = getattr(other, attr_name)
                    if attr is not None and attr != other_attr:
                        result = False
                        break
                # 'microsecond' is optional
                if result and hasattr(other, "microsecond"):
                    attr = self.microsecond
                    other_attr = other.microsecond
                    if attr is not None and attr != other_attr:
                        result = False

            except AttributeError:
                result = other.__eq__(self)
                if result is NotImplemented:
                    # Equality is undefined between these objects.  We don't
                    # want Python to fall back to the default `object`
                    # behaviour (which compares using object IDs), so we raise
                    # an exception here instead.
                    fmt = "unable to compare PartialDateTime with {}"
                    raise TypeError(fmt.format(type(other)))

        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result
