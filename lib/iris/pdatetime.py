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
