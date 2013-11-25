import collections
import netcdftime
import datetime
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

#
if __name__ == '__main__':
    from iris.pdatetime import PartialDateTime
    p = PartialDateTime(year=2010)

    d = datetime.datetime(2011, 1, 1)
    print p > d, d <= p
    print p < d, d >= p

#    print operator.gt(p, d)
#    print operator.gt(d, p)

    import iris
    cell = iris.coords.Cell(d)
    print cell > p, p <= cell
    print cell < p, p >= cell
#
#
#    # Interfaces for time constraints:
#
#    # Extract the 12th hour only (point and bound comparison)
#    cube.extract(iris.Constraint(time=PartialDateTime(hour=12)))
#    cube.extract(iris.Constraint(time=lambda cell: cell.point.hour == 12 and \
#                                                   np.all([dt.hour == 12 for dt in cell.bound])))
#
#    # Extract the 12th hour only (point only comparison)
#    # NOTE: First needs implementing, second needs a netcdftime bug fix.
#    cube.extract(iris.Constraint(time__point=PartialDateTime(hour=12)))
#    cube.extract(iris.Constraint(time=lambda cell: cell.point == PartialDateTime(hour=12)))
#    cube.extract(iris.Constraint(time=lambda cell: cell.point.hour == 12))
#
#    # Extract from 17th January to 30th February for any year in the 360 day
#    # calendar (calendar need only be specified once)
#    jan_17th = PartialDateTime(month=1, day=17, calendar='360day')
#    feb_30th = PartialDateTime(month=2, day=30)
#    cube.extract(iris.Constraint(time=lambda cell: jan_17th < cell <= feb_30th))
#
#    # First day of month for the Spring and Summer (climatological) seasons. With an hour of 12.
#    cube.extract(iris.Constraint(time=lambda cell: (cell.point.month in [3, 4, 5, 6, 7, 8]) and \
#                                                   (cell.point.day == 1 and cell.point.hour == 12)))
#
#    # Non climatological seasons (argh...)
#    spring_start = PartialDateTime(month=3, day=21, calendar='gregorian')
#    autumn_start = PartialDateTime(month=9, day=21)
#    spring_summer = iris.Constraint(time=lambda cell: spring_start < cell < autumn_start)
#    autumn_winter = iris.Constraint(time=lambda cell: autumn_start < cell < spring_start)
#    cube.extract(spring_summer & iris.Constraint(time=PartialDateTime(day=1, hour=12)))
#
#
    cube = iris.load_cube(iris.sample_data_path('A1B_north_america.nc'))
    gt_2010 = cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=2010) < cell <= PartialDateTime(year=2060)))
    print gt_2010.coord('time')
#
#
##    print gt_2010.coord('time')
#    #p = PartialDateTime(hour=10)
#    #t = datetime.time(21, 00)
#    #
#    #print p > t
#    #print p < t
#    #
#    #n = netcdftime.datetime(2010, 2, 30)
#    #n.calendar = 'foobar'
#    #
#    #p = PartialDateTime(year=2010, calendar='360day')
#    ## XXX Watch out for partial datetime comparison of eq... Bug in netcdftime.
#    #print p == n
#    #print p != n
