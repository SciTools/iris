# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Stopgap testing for pint units.

So far, only a few specific things are tested.
"""

from datetime import datetime

import cf_units
import numpy as np
import pytest

import iris.common.mixin
from iris.common.units import PintUnit


def test_num2date():
    unit = PintUnit("days since 1970-01-01")
    vals = np.array([1.0, 2])
    result = unit.num2date(vals)
    assert np.all(result == [datetime(1970, 1, 2), datetime(1970, 1, 3)])


def test_date2num():
    unit = PintUnit("days since 1970-01-01")
    vals = np.array([datetime(1970, 1, 2), datetime(1970, 1, 3)])
    result = unit.date2num(vals)
    assert np.all(result == [1.0, 2])


def test_nounit_eq():
    unit = PintUnit("m")
    assert unit != "no_unit"


def test_calendar():
    unit = PintUnit("days since 1970-01-01", calendar="360_day")
    # NOTE: no <>, due to "backwards compatibility" for assert_CDL
    # TODO: remove the PintUnit._REPR_NO_LTGT
    assert repr(unit) == "Unit('days since 1970-01-01', calendar='360_day')"
    # TODO: should really add the calendar to the string format
    #   I think this is a bit horrible,
    #   .. but it is cf_units behaviour + currently required for correct netcdf saving
    #   it also means that calendar is not checked in unit/string eq (!!!)
    assert str(unit) == "days since 1970-01-01"


_UNKNOWN_NAMES = iris.common.units.PintUnit._IRIS_EXTRA_CATEGORIES["unknown"]
_NOUNIT_NAMES = iris.common.units.PintUnit._IRIS_EXTRA_CATEGORIES["no_unit"]


class TestFromUnit:
    """Test PintUnit creation from various sources."""

    def test_none_unknown(self):
        unit = PintUnit.from_unit(None)
        assert unit.category == "unknown"
        assert unit.calendar is None
        assert unit == "unknown"

    @pytest.mark.parametrize("name", _UNKNOWN_NAMES)
    def test_str_unknown(self, name):
        unit = PintUnit.from_unit(None)
        assert unit.category == "unknown"
        assert unit.calendar is None
        assert all(unit == form for form in _UNKNOWN_NAMES)  # string equivalence

    def test_cfunits_unknown(self):
        cfunit = cf_units.Unit(None)
        unit = PintUnit.from_unit(None)
        assert unit.is_unknown()

    @pytest.mark.parametrize("name", _NOUNIT_NAMES)
    def test_str_nounit(self, name):
        unit = PintUnit.from_unit(name)
        assert unit.category == "no_unit"
        assert unit.calendar is None
        assert all(unit == form for form in _NOUNIT_NAMES)  # string equivalence

    def test_cfunits_nounit(self):
        cfunit = cf_units.Unit("no_unit")
        unit = PintUnit.from_unit(cfunit)
        assert unit.is_no_unit()

    def test_str(self):
        unit = PintUnit.from_unit("m")
        assert unit == "metres"  # string equivalence
        assert unit.calendar is None
        assert unit.category == "regular"

    def test_cfunits(self):
        cfunit = cf_units.Unit("m")
        unit = PintUnit.from_unit(cfunit)
        assert unit == "metre"

    def test_str_date(self):
        unit = PintUnit.from_unit("days since 1970-01-01")
        assert unit == "days since 1970-01-01"
        assert unit.category == "regular"
        assert unit.is_datelike()
        assert unit.calendar == "standard"

    @pytest.mark.skip("from_unit does not support calendar (yet?)")
    def test_str_date_calendar(self):
        unit = PintUnit.from_unit("days since 1970-01-01", calendar="360_day")
        # YUCK!! cf_units compatibility
        # TODO: this needs to change
        assert unit == "days since 1970-01-01"
        assert unit.category == "regular"
        assert unit.is_datelike()
        assert unit.calendar == "360_day"

    def test_cfunits_date(self):
        cfunit = cf_units.Unit("hours since 1800-03-09 11:11")
        unit = PintUnit.from_unit(cfunit)
        # NB time ref is reproduced as-is
        # TODO: should get normalised
        assert unit == "hours since 1800-03-09 11:11"
        assert unit.is_datelike()
        assert unit.calendar == "standard"

    def test_cfunits_date_calendar(self):
        cfunit = cf_units.Unit("hours since 1800-03-09 11:11", calendar="365_day")
        unit = PintUnit.from_unit(cfunit)
        # NB time ref is reproduced as-is
        # TODO: should get normalised
        assert unit == "hours since 1800-03-09 11:11"
        assert unit.is_datelike()
        assert unit.calendar == "365_day"
