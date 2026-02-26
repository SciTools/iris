from datetime import datetime

import numpy as np

from iris.common.mixin import CfpintUnit


def test_num2date():
    unit = CfpintUnit("days since 1970-01-01")
    vals = np.array([1.0, 2])
    result = unit.num2date(vals)
    assert np.all(result == [datetime(1970, 1, 2), datetime(1970, 1, 3)])


def test_date2num():
    unit = CfpintUnit("days since 1970-01-01")
    vals = np.array([datetime(1970, 1, 2), datetime(1970, 1, 3)])
    result = unit.date2num(vals)
    assert np.all(result == [1.0, 2])


def test_nounit_eq():
    unit = CfpintUnit("m")
    assert unit != "no_unit"


def test_calendar():
    unit = CfpintUnit("days since 1970-01-01", calendar="360_day")
    assert repr(unit) == "<Unit('days since 1970-01-01', calendar='360_day')>"
    # TODO: should really add the calendar to the string format
    #   I think this is a bit horrible,
    #   .. but it is cf_units behaviour + currently required for correct netcdf saving
    #   it also means that calendar is not checked in unit/string eq (!!!)
    assert str(unit) == "days since 1970-01-01"
